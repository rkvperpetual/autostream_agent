"""
state.py — LangGraph State Definition for AutoStream Agent

Defines the shared state object that flows through every node in the graph.
All fields are persisted across conversation turns automatically by LangGraph.
"""

from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    """
    Shared state that is passed between every node in the LangGraph graph.

    Fields:
        messages       : Full conversation history as a list of dicts
                         Each dict: {"role": "user" | "assistant", "content": "..."}
        intent         : Classified intent of the latest user message
                         One of: "greeting", "inquiry", "high_intent", "unknown"
        lead_name      : Collected name of the lead (None if not yet provided)
        lead_email     : Collected email of the lead (None if not yet provided)
        lead_platform  : Collected creator platform (None if not yet provided)
        lead_captured  : True once mock_lead_capture() has been successfully called
        response       : The latest assistant response string (sent back to the user)
    """

    messages: List[dict]
    intent: str
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    response: str