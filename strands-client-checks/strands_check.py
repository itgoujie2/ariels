"""Reachability check using AWS's own first-party A2A client
(`strands.agent.a2a_agent.A2AAgent`) -- see strands_prober.py's
docstring for the full design rationale.

Deliberately minimal: does one basic message get a real reply? Mirrors
the other native-client layers' shallow-check scope.
"""

from __future__ import annotations

from strands.agent.a2a_agent import A2AAgent

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    agent = A2AAgent(endpoint=base_url, timeout=30)
    try:
        result = await agent.invoke_async(PROBE_TEXT)
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}

    texts = [c.get("text") for c in (result.message.get("content") or []) if c.get("text")]
    return {"reachable": True, "reply": "\n".join(texts) if texts else None, "error": None}
