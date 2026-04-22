"""
graph.py — LangGraph State Machine for AutoStream Agent

Defines the full conversation graph:
  START
    └─► intent_node
          ├─► greeting_node      (intent = "greeting")
          ├─► rag_node           (intent = "inquiry" and no partial lead data)
          └─► lead_collection_node (intent = "high_intent" OR partial lead in progress)
                ├─► tool_execution_node  (all three fields collected)
                └─► END                  (still collecting)

State persists across every turn — LangGraph carries it through automatically.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    intent_node,
    greeting_node,
    rag_node,
    lead_collection_node,
    tool_execution_node,
)


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """
    Route based on the classified intent AND the current lead collection progress.

    Priority rules:
    1. If ANY lead field has already been collected (i.e. mid-collection),
       always continue collecting — regardless of intent.
    2. If intent is high_intent, start collection.
    3. If intent is greeting, respond with a greeting.
    4. Default: send to RAG for product/pricing answers.
    """
    in_collection = any([
        state.get("lead_name"),
        state.get("lead_email"),
        state.get("lead_platform"),
    ])

    if in_collection:
        return "collect_lead"

    if state["intent"] == "greeting":
        return "greeting"

    if state["intent"] == "high_intent":
        return "collect_lead"

    return "rag"  # Default — handles "inquiry" and "unknown"


def route_after_lead_collection(state: AgentState) -> str:
    """
    After lead_collection_node runs, check if all three fields are now present.

    - If yes → fire the mock tool via tool_execution_node.
    - If no  → end this turn (wait for user to provide the next field).
    """
    all_collected = all([
        state.get("lead_name"),
        state.get("lead_email"),
        state.get("lead_platform"),
    ])
    return "capture_tool" if all_collected else END


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Returns:
        A compiled LangGraph app ready to invoke with invoke(state).
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("intent",       intent_node)
    graph.add_node("greeting",     greeting_node)
    graph.add_node("rag",          rag_node)
    graph.add_node("collect_lead", lead_collection_node)
    graph.add_node("capture_tool", tool_execution_node)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("intent")

    # ── Conditional routing after intent classification ───────────────────
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "greeting":     "greeting",
            "rag":          "rag",
            "collect_lead": "collect_lead",
        },
    )

    # ── Terminal edges for greeting and RAG (single-turn responses) ───────
    graph.add_edge("greeting", END)
    graph.add_edge("rag",      END)

    # ── Lead collection may loop or proceed to tool execution ─────────────
    graph.add_conditional_edges(
        "collect_lead",
        route_after_lead_collection,
        {
            "capture_tool": "capture_tool",
            END:            END,
        },
    )

    # ── Tool execution always terminates the graph for that turn ─────────
    graph.add_edge("capture_tool", END)

    return graph.compile()