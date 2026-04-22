"""
rag.py — Retrieval-Augmented Generation (RAG) Pipeline

Loads the local AutoStream knowledge base (JSON) and retrieves
relevant sections based on the user's query using keyword matching.

In a production system this would use a vector store (FAISS / Chroma),
but for this assignment the KB is small enough for keyword-based retrieval.
"""

import json
from pathlib import Path


# ─── Load Knowledge Base ────────────────────────────────────────────────────

def load_knowledge_base() -> dict:
    """Load and return the full knowledge base from the JSON file."""
    kb_path = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Keyword Groups ──────────────────────────────────────────────────────────

PRICING_KEYWORDS = {
    "price", "pricing", "plan", "plans", "cost", "how much", "subscription",
    "basic", "pro", "feature", "features", "resolution", "video", "caption",
    "support", "4k", "720", "unlimited", "month", "monthly", "cheap", "afford",
    "expensive", "tier", "package", "deal",
}

POLICY_KEYWORDS = {
    "refund", "policy", "policies", "cancel", "cancellation", "trial", "free trial",
    "billing", "annual", "yearly", "upgrade", "downgrade", "access", "storage",
    "credit card", "payment", "charge",
}

FAQ_KEYWORDS = {
    "format", "mp4", "mov", "avi", "mkv", "platform", "youtube", "instagram",
    "tiktok", "twitter", "facebook", "export", "type", "kind", "creator",
    "content", "tutorial", "vlog", "podcast", "reels", "short",
}


# ─── Retrieval Function ──────────────────────────────────────────────────────

def get_relevant_context(query: str) -> str:
    """
    Retrieve relevant knowledge base sections based on the user's query.

    Strategy:
    1. Tokenise the query into lowercase words.
    2. Check against keyword groups to decide which KB sections are relevant.
    3. Always include product header; add plans / policies / FAQs as matched.
    4. Fall back to the full KB if nothing specific matches.

    Args:
        query: The latest user message.

    Returns:
        A formatted string of relevant KB content to use as LLM context.
    """
    kb = load_knowledge_base()
    query_lower = query.lower()
    tokens = set(query_lower.split())

    context_parts: list[str] = []

    # ── Product overview (always included) ────────────────────────────────
    context_parts.append(f"## Product: {kb['product']}")
    context_parts.append(f"Tagline: {kb['tagline']}")
    context_parts.append(f"Description: {kb['description']}\n")

    # ── Decide which sections to include ─────────────────────────────────
    include_pricing = bool(tokens & PRICING_KEYWORDS) or any(
        kw in query_lower for kw in PRICING_KEYWORDS if " " in kw
    )
    include_policy = bool(tokens & POLICY_KEYWORDS) or any(
        kw in query_lower for kw in POLICY_KEYWORDS if " " in kw
    )
    include_faq = bool(tokens & FAQ_KEYWORDS) or any(
        kw in query_lower for kw in FAQ_KEYWORDS if " " in kw
    )

    # If nothing matched, show everything
    if not any([include_pricing, include_policy, include_faq]):
        include_pricing = include_policy = include_faq = True

    # ── Pricing / Plans ────────────────────────────────────────────────────
    if include_pricing:
        context_parts.append("## Pricing Plans")
        for plan in kb["plans"]:
            context_parts.append(f"\n### {plan['name']} — {plan['price']}")
            context_parts.append("Features:")
            for feature in plan["features"]:
                context_parts.append(f"  - {feature}")

    # ── Policies ──────────────────────────────────────────────────────────
    if include_policy:
        context_parts.append("\n## Company Policies")
        for policy in kb["policies"]:
            context_parts.append(f"  - {policy}")

    # ── FAQs ──────────────────────────────────────────────────────────────
    if include_faq:
        context_parts.append("\n## Frequently Asked Questions")
        for faq in kb["faqs"]:
            context_parts.append(f"\nQ: {faq['question']}")
            context_parts.append(f"A: {faq['answer']}")

    return "\n".join(context_parts)


# ─── Helper: format full KB as plain text ────────────────────────────────────

def get_full_context() -> str:
    """Return the entire knowledge base as formatted text (used as fallback)."""
    return get_relevant_context("")