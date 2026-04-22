"""
nodes.py — LangGraph Node Functions for AutoStream Agent

Compatible with: Claude (Anthropic), Gemini / Gemma (Google), GPT (OpenAI)

Each node is a pure function: (AgentState) → AgentState.
Nodes are connected in graph.py to form the full conversation pipeline.

Node overview:
  intent_node          → Classifies user intent (greeting / inquiry / high_intent)
  greeting_node        → Handles casual greetings
  rag_node             → Answers product/pricing questions using RAG
  lead_collection_node → Collects name, email, and platform step by step
  tool_execution_node  → Calls mock_lead_capture() once all fields are ready
"""

import json
import re
import os

from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from .rag import get_relevant_context
from .tools import mock_lead_capture


# ─── Universal Text Extractor ─────────────────────────────────────────────────

def extract_text(response) -> str:
    """
    Safely extract plain text from an LLM response.

    Handles all response formats across providers:
      - Claude (Anthropic)  : response.content is a plain string
      - Gemini / Gemma      : response.content is a list of dicts
                              e.g. [{'type': 'thinking', 'thinking': '...'},
                                    {'type': 'text',    'text': '...'}]
      - OpenAI              : response.content is a plain string
      - Any other provider  : falls back to str(content)

    Only 'text' type blocks are included. 'thinking' / 'reasoning' blocks
    are always skipped so internal chain-of-thought never leaks to the user.
    """
    content = response.content

    # ── List of blocks (Gemini / Gemma style) ─────────────────────────────
    if isinstance(content, list):
        text_parts = []

        for block in content:
            # Dict-style block: {'type': 'text', 'text': '...'}
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            # Object-style block: block.type == 'text', block.text == '...'
            elif getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))

        # Fallback: if nothing matched the 'text' filter,
        # grab anything that has a text value (avoids silent empty responses)
        if not text_parts:
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
                elif hasattr(block, "text"):
                    text_parts.append(str(block.text))

        return "".join(text_parts).strip()

    # ── Plain string (Claude / OpenAI style) ──────────────────────────────
    return str(content).strip()


# ─── LLM Initialisation ──────────────────────────────────────────────────────

def get_llm():
    """
    Initialise and return the configured LLM.

    Reads the LLM_PROVIDER environment variable (set in .env) to decide
    which provider to use. Defaults to Google Gemini if not set.

    Supported values for LLM_PROVIDER:
      gemini   → Google Gemini 1.5 Flash      (default)
      claude   → Anthropic Claude 3 Haiku
      openai   → OpenAI GPT-4o-mini

    The corresponding API key must also be set in .env.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.3,
            max_tokens=512,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=512,
        )

    else:
        # Default → Gemini (also covers gemma / any google model)
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite-preview")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            max_tokens=512,
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_history(messages: list, last_n: int = 8) -> str:
    """
    Format the last N conversation turns as a readable string for prompts.

    Args:
        messages : Full message list
        last_n   : How many recent messages to include
    """
    recent = messages[-last_n:] if len(messages) > last_n else messages
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def clean_json(raw: str) -> str:
    """Strip markdown code fences from an LLM JSON response."""
    return re.sub(r"```json|```", "", raw).strip()


# ─── Node 1: Intent Classification ───────────────────────────────────────────

def intent_node(state: AgentState) -> AgentState:
    """
    Classify the intent of the latest user message.

    Intents:
      greeting    → Hi, hello, hey, good morning, etc.
      inquiry     → Questions about product, pricing, features, policies
      high_intent → User wants to sign up, try, purchase, or subscribe

    Updates state["intent"] and returns the updated state.
    Does NOT generate a user-facing response.
    """
    last_message = state["messages"][-1]["content"]
    history = format_history(state["messages"][:-1])

    system = (
        "You are an intent classifier for AutoStream, a SaaS video editing platform.\n\n"
        "Classify the user's LATEST message into EXACTLY ONE of:\n"
        "  greeting    — Simple greetings (hi, hello, hey, good morning, etc.)\n"
        "  inquiry     — Questions about features, pricing, plans, policies, how it works\n"
        "  high_intent — User clearly wants to sign up, try, purchase, or subscribe\n\n"
        "Conversation so far:\n"
        f"{history}\n\n"
        "Rules:\n"
        "- Respond with ONLY the single intent word. No punctuation. No explanation.\n"
        "- If unsure, default to: inquiry"
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Latest user message: {last_message}"),
    ])

    raw = extract_text(response).lower().split()
    intent = raw[0] if raw else "inquiry"

    if intent not in ("greeting", "inquiry", "high_intent"):
        intent = "inquiry"

    return {**state, "intent": intent}


# ─── Node 2: Greeting ─────────────────────────────────────────────────────────

def greeting_node(state: AgentState) -> AgentState:
    """
    Respond to a casual greeting.

    Introduces AutoStream briefly and invites the user to ask
    about features or pricing.
    """
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a friendly and enthusiastic sales assistant for AutoStream, "
            "an AI-powered automated video editing SaaS platform for content creators.\n\n"
            "The user just greeted you. Write a warm 2-3 sentence reply that:\n"
            "1. Greets them back\n"
            "2. Briefly says what AutoStream does\n"
            "3. Invites them to ask about features or pricing\n\n"
            "Tone: friendly, professional, not pushy."
        )),
        HumanMessage(content=state["messages"][-1]["content"]),
    ])

    text = extract_text(response)
    updated_messages = state["messages"] + [{"role": "assistant", "content": text}]
    return {**state, "messages": updated_messages, "response": text}


# ─── Node 3: RAG-Powered Answer ───────────────────────────────────────────────

def rag_node(state: AgentState) -> AgentState:
    """
    Answer product/pricing questions using the local knowledge base (RAG).

    Steps:
      1. Retrieve relevant KB sections based on the user's query
      2. Pass retrieved context + conversation history to the LLM
      3. LLM generates a grounded, accurate answer
    """
    last_message = state["messages"][-1]["content"]
    context = get_relevant_context(last_message)
    history = format_history(state["messages"][:-1])

    system = (
        "You are a knowledgeable and friendly sales assistant for AutoStream, "
        "an AI-powered automated video editing SaaS platform for content creators.\n\n"
        "────────────────────────────────────────\n"
        "KNOWLEDGE BASE (use this to answer):\n"
        "────────────────────────────────────────\n"
        f"{context}\n"
        "────────────────────────────────────────\n\n"
        "Conversation so far:\n"
        f"{history}\n\n"
        "Rules:\n"
        "- Answer ONLY from the knowledge base above. Do not make up information.\n"
        "- Be concise, clear, and helpful.\n"
        "- If you cannot find the answer, say: "
        "'I don't have that specific information, but our team can help — "
        "would you like to connect with us?'\n"
        "- If the user seems interested, naturally mention the Pro plan offers the best value."
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=last_message),
    ])

    text = extract_text(response)
    updated_messages = state["messages"] + [{"role": "assistant", "content": text}]
    return {**state, "messages": updated_messages, "response": text}


# ─── Node 4: Lead Collection ──────────────────────────────────────────────────

def lead_collection_node(state: AgentState) -> AgentState:
    """
    Collect lead information step by step: name → email → platform.

    On each turn:
      1. Try to extract any of the three fields from the user's message.
      2. Determine which field is still missing.
      3. Ask for the next missing field conversationally.

    Never asks for more than one field at a time.
    Once all three fields are collected, graph routes to tool_execution_node.
    """
    last_message = state["messages"][-1]["content"]
    updated = dict(state)

    # ── Step 1: Extract info from the current message ─────────────────────
    extract_prompt = (
        f"Extract any of the following from the message below.\n\n"
        f"Message: \"{last_message}\"\n\n"
        f"Already collected:\n"
        f"  name     = {state.get('lead_name') or 'NOT YET'}\n"
        f"  email    = {state.get('lead_email') or 'NOT YET'}\n"
        f"  platform = {state.get('lead_platform') or 'NOT YET'}\n\n"
        "Return ONLY a valid JSON object with keys: name, email, platform.\n"
        "Use null for anything not clearly present in the message.\n"
        "For platform, accept: YouTube, Instagram, TikTok, Twitter, "
        "Facebook, LinkedIn, Twitch, or Other.\n"
        "Example: {\"name\": \"Priya Sharma\", \"email\": null, \"platform\": \"YouTube\"}"
    )

    llm = get_llm()
    extract_resp = llm.invoke([
        SystemMessage(content=(
            "You are a precise data extraction assistant. "
            "Extract only what is explicitly stated. "
            "Return valid JSON only. No preamble. No explanation. No markdown."
        )),
        HumanMessage(content=extract_prompt),
    ])

    try:
        extracted = json.loads(clean_json(extract_text(extract_resp)))
        if extracted.get("name") and not updated.get("lead_name"):
            updated["lead_name"] = str(extracted["name"]).strip()
        if extracted.get("email") and not updated.get("lead_email"):
            updated["lead_email"] = str(extracted["email"]).strip()
        if extracted.get("platform") and not updated.get("lead_platform"):
            updated["lead_platform"] = str(extracted["platform"]).strip()
    except (json.JSONDecodeError, TypeError):
        pass  # Extraction failed — agent will ask for the field again

    # ── Step 2: Decide what to ask next ──────────────────────────────────
    already_have = {
        "name":     updated.get("lead_name"),
        "email":    updated.get("lead_email"),
        "platform": updated.get("lead_platform"),
    }

    if not already_have["name"]:
        ask_instruction = (
            "The user wants to sign up for AutoStream Pro. "
            "Acknowledge their interest enthusiastically, then ask for their full name. "
            "1-2 sentences. Be warm and natural."
        )
    elif not already_have["email"]:
        ask_instruction = (
            f"You already have their name: {already_have['name']}. "
            "Thank them and ask for their email address. "
            "1-2 sentences."
        )
    elif not already_have["platform"]:
        ask_instruction = (
            f"You have their name ({already_have['name']}) "
            f"and email ({already_have['email']}). "
            "Now ask which creator platform they primarily publish on "
            "(e.g. YouTube, Instagram, TikTok, etc.). "
            "1-2 sentences."
        )
    else:
        # All fields collected — route will pick this up and go to tool_execution
        summary = (
            f"Great! I have everything I need:\n"
            f"  Name     : {already_have['name']}\n"
            f"  Email    : {already_have['email']}\n"
            f"  Platform : {already_have['platform']}"
        )
        updated["response"] = summary
        updated["messages"] = state["messages"] + [
            {"role": "assistant", "content": summary}
        ]
        return updated

    # ── Step 3: Generate the conversational ask ───────────────────────────
    response = llm.invoke([
        SystemMessage(content=(
            f"You are a friendly sales assistant for AutoStream. {ask_instruction}"
        )),
        HumanMessage(content=last_message),
    ])

    text = extract_text(response)
    updated["response"] = text
    updated["messages"] = state["messages"] + [{"role": "assistant", "content": text}]
    return updated


# ─── Node 5: Tool Execution ───────────────────────────────────────────────────

def tool_execution_node(state: AgentState) -> AgentState:
    """
    Called ONLY after all three lead fields are confirmed.

    1. Calls mock_lead_capture() with the collected information.
    2. Generates a warm confirmation message for the user.
    3. Sets lead_captured = True to signal the conversation is complete.
    """
    # ── Call the mock tool ────────────────────────────────────────────────
    mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"],
    )

    # ── Generate confirmation message ─────────────────────────────────────
    llm = get_llm()
    confirm_prompt = (
        f"A new user just signed up for AutoStream Pro!\n\n"
        f"  Name     : {state['lead_name']}\n"
        f"  Email    : {state['lead_email']}\n"
        f"  Platform : {state['lead_platform']}\n\n"
        "Write a warm, enthusiastic 2-3 sentence confirmation message that:\n"
        "1. Confirms their registration is complete\n"
        f"2. Mentions they will receive a welcome email at {state['lead_email']}\n"
        f"3. Wishes them success creating content on {state['lead_platform']}\n"
        "Tone: celebratory but professional."
    )

    response = llm.invoke([
        SystemMessage(content="You are a friendly sales assistant for AutoStream."),
        HumanMessage(content=confirm_prompt),
    ])

    text = extract_text(response)
    updated_messages = state["messages"] + [{"role": "assistant", "content": text}]

    return {
        **state,
        "messages":      updated_messages,
        "response":      text,
        "lead_captured": True,
    }