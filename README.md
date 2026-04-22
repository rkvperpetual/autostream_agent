# AutoStream AI Agent
### Social-to-Lead Agentic Workflow — ML Intern Assignment (ServiceHive / Inflx)

> A production-quality Conversational AI Agent that answers product questions, detects high-intent users, and captures qualified leads — built with **LangGraph + Gemini 2.5 Flash**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Project Structure](#3-project-structure)
4. [How to Run Locally](#4-how-to-run-locally)
5. [Architecture Explanation](#5-architecture-explanation)
6. [Conversation Flow](#6-conversation-flow)
7. [WhatsApp Deployment via Webhooks](#7-whatsapp-deployment-via-webhooks)
8. [Switching the LLM](#8-switching-the-llm)
9. [Evaluation Checklist](#9-evaluation-checklist)

---

## 1. Project Overview

**AutoStream** is a fictional SaaS platform for automated video editing. This agent acts as an AI-powered sales assistant that:

- Understands what the user wants (intent classification)
- Answers product and pricing questions from a local knowledge base (RAG)
- Identifies when a user is ready to sign up (high-intent detection)
- Collects lead details (name, email, platform) conversationally
- Calls a mock lead-capture API once all information is gathered

---

## 2. Features

| Capability | Implementation |
|---|---|
| Intent Classification | LLM-based classifier (greeting / inquiry / high_intent) |
| RAG Knowledge Retrieval | Local JSON knowledge base + keyword-based retrieval |
| Multi-turn Memory | LangGraph `AgentState` persists across 5–6+ turns |
| Lead Collection | Step-by-step extraction with LLM-powered field parsing |
| Tool Execution Guard | `mock_lead_capture()` fires **only** when all 3 fields present |
| Conversation Routing | Conditional edges in LangGraph state machine |

---

## 3. Project Structure

```
autostream-agent/
│
├── knowledge_base/
│   └── autostream_kb.json      # Local KB: pricing, features, policies, FAQs
│
├── agents/
│   ├── __init__.py
│   ├── state.py                # AgentState TypedDict (shared across all nodes)
│   ├── rag.py                  # RAG pipeline: load KB → retrieve context
│   ├── tools.py                # mock_lead_capture() function
│   ├── nodes.py                # All LangGraph node functions
│   └── graph.py                # StateGraph definition + routing logic
│
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example                # Template for environment variables
└── README.md
```

---

## 4. How to Run Locally

### Prerequisites

- Python 3.9 or higher
- An Google API key ([get one here](https://aistudio.google.com/api-keys))

### Step 1 — Clone the Repository

```bash
git clone https://github.com/rkvperpetual/autostream-agent.git
cd autostream-agent
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Activate:
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key
```

### Step 5 — Run the Agent

```bash
python main.py
```

You should see the welcome banner and can start chatting immediately.

### Example Session

```
You: Hi there!
Assistant: Hey! Welcome to AutoStream 👋 I'm your AI assistant...

You: What plans do you offer?
Assistant: AutoStream offers two plans:
  • Basic — $29/month: 10 videos, 720p, email support
  • Pro   — $79/month: Unlimited videos, 4K, AI captions, 24/7 support...

You: The Pro plan sounds great, I want to sign up for my YouTube channel.
Assistant: Awesome! I'd love to get you started. What's your full name?

You: Priya Sharma
Assistant: Thanks Priya! What's your email address?

You: priya@example.com
Assistant: Perfect! Which platform do you mainly create content on?

You: YouTube
Assistant: 🎉 You're all set, Priya! Welcome to AutoStream Pro...

✅ LEAD CAPTURED SUCCESSFULLY
   Name     : Priya Sharma
   Email    : priya@example.com
   Platform : YouTube
```

---

## 5. Architecture Explanation

### Why LangGraph?

LangGraph was chosen over AutoGen for this project because it offers **explicit, inspectable state management** — critical for a multi-step lead collection workflow where the agent must remember what it has already asked across turns.

Unlike a simple memory buffer that just appends messages, LangGraph models the conversation as a **directed graph** where each node is a pure function `(AgentState) → AgentState`. This means:

- State (intent, collected lead fields, captured flag) is typed and explicit
- Routing logic is separated from response logic — easy to test and extend
- Adding new steps (e.g. "ask for company name") only requires a new node and edge

### How State is Managed

All state lives in a single `AgentState` TypedDict that flows through every node:

```
AgentState {
  messages      : full conversation history (list of dicts)
  intent        : latest classified intent
  lead_name     : collected or None
  lead_email    : collected or None
  lead_platform : collected or None
  lead_captured : True once tool fires
  response      : latest assistant message
}
```

LangGraph carries this object between nodes automatically. There is no external database — the state is kept in memory for the duration of the session (5–6+ turns easily supported).

### Node Execution Flow

```
User Message
     │
     ▼
┌──────────────┐
│ intent_node  │  ← Classifies: greeting / inquiry / high_intent
└──────┬───────┘
       │
       ├─ greeting   ──► greeting_node  ──► END
       │
       ├─ inquiry    ──► rag_node       ──► END
       │                 (RAG-powered answer from local KB)
       │
       └─ high_intent ──► lead_collection_node
          (or any                │
          partial lead           ├─ fields missing ──► END (wait for next turn)
          already in             │
          state)                 └─ all collected  ──► tool_execution_node ──► END
                                                        (calls mock_lead_capture)
```

### RAG Pipeline

The RAG pipeline avoids a vector store (unnecessary for a small KB) and instead uses **keyword-based section retrieval**:

1. User query is tokenised into lowercase words
2. Tokens are matched against predefined keyword groups (pricing, policies, FAQs)
3. Matching sections from `autostream_kb.json` are assembled into a context string
4. Context is injected into the LLM system prompt alongside conversation history

This approach is fast, deterministic, and easy to extend by adding new keyword groups or KB sections.

---

## 6. Conversation Flow

The agent handles three distinct paths:

### Path A — Greeting
```
User: "Hi!"
→ intent_node classifies: greeting
→ greeting_node returns a warm intro with an invitation to ask about plans
```

### Path B — Product Inquiry
```
User: "What's the difference between Basic and Pro?"
→ intent_node classifies: inquiry
→ rag_node retrieves pricing section from KB
→ LLM generates a grounded answer
```

### Path C — Lead Capture (Multi-turn)
```
Turn 1: "I want to try the Pro plan for my channel."
→ intent: high_intent → lead_collection_node → asks for name

Turn 2: "My name is Priya Sharma"
→ intent: (anything) → still in collection → lead_collection_node extracts name → asks for email

Turn 3: "priya@example.com"
→ lead_collection_node extracts email → asks for platform

Turn 4: "YouTube"
→ lead_collection_node extracts platform → all 3 collected
→ routes to tool_execution_node → calls mock_lead_capture()
```

---

## 7. WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following integration architecture would be used:

### Option A — Twilio WhatsApp API (Recommended for Quick Deployment)

```
WhatsApp User
     │  (sends message)
     ▼
Twilio WhatsApp API
     │  (POST webhook with message body)
     ▼
FastAPI / Flask Webhook Server  ──► AutoStream LangGraph Agent
     │                                  │
     │  (runs graph.invoke(state))       │
     │◄──────────────────────────────────┘
     │  (agent response)
     ▼
Twilio API  ──► WhatsApp User (reply)
```

### Implementation Steps

**1. Create a FastAPI webhook endpoint:**

```python
from fastapi import FastAPI, Form
from twilio.rest import Client
from agent.graph import build_graph
from agent.state import AgentState

app = FastAPI()
sessions = {}   # In-memory session store; use Redis in production

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

@app.post("/webhook")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    user_id = From  # Phone number as session key

    # Load or create session state
    if user_id not in sessions:
        sessions[user_id] = {
            "messages": [], "intent": "unknown",
            "lead_name": None, "lead_email": None,
            "lead_platform": None, "lead_captured": False, "response": ""
        }

    state = sessions[user_id]
    state["messages"].append({"role": "user", "content": Body})

    graph = build_graph()
    result = graph.invoke(state)
    sessions[user_id] = result

    # Reply via Twilio
    twilio_client.messages.create(
        from_="whatsapp:+14155238886",
        to=From,
        body=result["response"]
    )
    return {"status": "ok"}
```

**2. Expose the server publicly using ngrok (for development):**

```bash
uvicorn webhook:app --port 8000
ngrok http 8000
```

**3. Configure Twilio:**
- Go to Twilio Console → Messaging → WhatsApp Sandbox
- Set the webhook URL to: `https://YOUR_NGROK_URL/webhook`

**4. Production Deployment:**
- Deploy the FastAPI app to **Railway**, **Render**, or **AWS Lambda**
- Replace in-memory `sessions` dict with **Redis** for persistent multi-user state
- Register a proper WhatsApp Business number via Meta or Twilio

### Option B — Meta Cloud API (Official)
For a production system, use the **Meta WhatsApp Business Cloud API** directly:
- Register at [developers.facebook.com](https://developers.facebook.com)
- Set up a webhook that receives `messages` events
- Respond via `POST https://graph.facebook.com/v18.0/{PHONE_ID}/messages`

---

## 8. Switching the LLM

The LLM is initialised in `agent/nodes.py` inside `get_llm()`. To switch providers:

### GPT-4o-mini (OpenAI)

```python
# 1. pip install langchain-openai
# 2. Set OPENAI_API_KEY in .env
from langchain_openai import ChatOpenAI
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
```

### Gemini 2.5 Flash (Google)

```python
# 1. pip install langchain-google-genai
# 2. Set GOOGLE_API_KEY in .env
from langchain_google_genai import ChatGoogleGenerativeAI
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
```

---

## 9. Evaluation Checklist

| Criterion | ✅ Status |
|---|---|
| Intent detection (greeting / inquiry / high_intent) | ✅ Implemented in `intent_node` |
| RAG pipeline using local knowledge base | ✅ `rag.py` + `autostream_kb.json` |
| State retained across 5–6 turns | ✅ `AgentState` via LangGraph |
| Lead collection: name, email, platform | ✅ `lead_collection_node` |
| Tool fires only after all 3 fields | ✅ `route_after_lead_collection` guard |
| `mock_lead_capture()` function | ✅ `agent/tools.py` |
| LangGraph used as framework | ✅ `agent/graph.py` |
| Gemini 2.5 Flash as LLM | ✅ `agent/nodes.py` |
| Clean code structure | ✅ Modular: state / rag / tools / nodes / graph |
| `requirements.txt` | ✅ |
| README with architecture + WhatsApp | ✅ This file |

---

## License

This project was built as part of the ServiceHive / Inflx ML Intern Assignment.