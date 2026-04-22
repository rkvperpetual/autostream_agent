"""
tools.py — Tool Definitions for AutoStream Agent

Contains the mock lead capture function that is triggered by the agent
ONLY after all three required fields (name, email, platform) are collected.
"""

import datetime


# ─── Mock Lead Capture API ────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture a qualified lead.

    In a real production system, this would:
    - POST to a CRM (HubSpot / Salesforce)
    - Send a welcome email via SendGrid
    - Notify the sales team via Slack

    Args:
        name     : Full name of the lead
        email    : Email address of the lead
        platform : Primary content creator platform (YouTube, Instagram, etc.)

    Returns:
        dict with status, message, and lead details
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Simulated API call output ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ✅  LEAD CAPTURED SUCCESSFULLY  ")
    print("=" * 55)
    print(f"  Name      : {name}")
    print(f"  Email     : {email}")
    print(f"  Platform  : {platform}")
    print(f"  Timestamp : {timestamp}")
    print("=" * 55 + "\n")

    return {
        "status": "success",
        "message": f"Lead captured for {name}",
        "timestamp": timestamp,
        "lead": {
            "name": name,
            "email": email,
            "platform": platform,
        },
    }