"""
main.py — AutoStream AI Agent Entry Point

Runs an interactive CLI chat loop that feeds user messages into
the LangGraph agent and prints its responses.

Usage:
    python main.py
"""

import os
from dotenv import load_dotenv
from agents.graph import build_graph
from agents.state import AgentState

# ─── Load environment variables from .env ────────────────────────────────────
load_dotenv()

# ─── Validate API key ─────────────────────────────────────────────────────────
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError(
        "\n❌  GOOGLE_API_KEY not found.\n"
        "    Please create a .env file based on .env.example and add your key.\n"
        "    See README.md for setup instructions."
    )


# ─── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║         AutoStream AI Sales Assistant                ║
║         Powered by Gemini 2.5 Flash + LangGraph        ║
╚══════════════════════════════════════════════════════╝
  Ask about pricing, features, or sign up for Pro!
  Type  'quit'  or  'exit'  to end the conversation.
══════════════════════════════════════════════════════
"""


# ─── Chat Loop ────────────────────────────────────────────────────────────────

def run_chat() -> None:
    """
    Main interactive chat loop.

    Maintains a single AgentState across all turns.
    On each user message:
      1. Appends message to state["messages"]
      2. Invokes the compiled LangGraph app
      3. Prints the agent's response
      4. Breaks when lead_captured = True or user types exit
    """
    print(BANNER)

    # Build the compiled graph once
    app = build_graph()

    # Initialise a fresh state
    state: AgentState = {
        "messages":      [],
        "intent":        "unknown",
        "lead_name":     None,
        "lead_email":    None,
        "lead_platform": None,
        "lead_captured": False,
        "response":      "",
    }

    while True:
        # ── Get user input ─────────────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAssistant: Thanks for stopping by! Goodbye 👋\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("\nAssistant: Thanks for chatting with AutoStream! "
                  "Feel free to come back anytime. Goodbye! 👋\n")
            break

        # ── Append user message to state ──────────────────────────────────
        state["messages"].append({"role": "user", "content": user_input})

        # ── Invoke the LangGraph app ──────────────────────────────────────
        try:
            result = app.invoke(state)
            state = result  # Persist updated state for next turn
        except Exception as exc:
            print(f"\nAssistant: I ran into an issue. Please try again.\n")
            print(f"[Debug] {type(exc).__name__}: {exc}\n")
            # Remove the failed user message so state stays clean
            state["messages"].pop()
            continue

        # ── Print assistant response ──────────────────────────────────────
        print(f"\nAssistant: {state['response']}\n")

        # ── End conversation after successful lead capture ─────────────────
        if state.get("lead_captured"):
            print("─" * 54)
            print("  🎉  Lead captured! Conversation complete.")
            print("─" * 54 + "\n")
            break


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_chat()