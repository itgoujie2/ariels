"""Reachability check using AG2's real, official, first-party A2A client
(`A2aRemoteAgent`, the `ag2[a2a]` extra) -- see ag2_prober.py's docstring
for the full design rationale.

Deliberately minimal: does the client resolve the card, and does sending
one basic message get a real reply? Mirrors the ADK and raw-SDK layers'
shallow-check scope for a fair, apples-to-apples comparison.
"""

from __future__ import annotations

from autogen.a2a.client import A2aRemoteAgent

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    try:
        agent = A2aRemoteAgent(url=base_url, name="ag2_check")
        ok, reply = await agent.a_generate_remote_reply(messages=[{"role": "user", "content": PROBE_TEXT}])
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}

    if not ok or reply is None:
        return {"reachable": False, "reply": None, "error": "no reply returned"}

    text = reply.get("content", "") if isinstance(reply, dict) else str(reply)
    return {"reachable": True, "reply": text, "error": None}
