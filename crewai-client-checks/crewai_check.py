"""Reachability check using CrewAI's own first-party A2A client
(`crewai.a2a.utils.delegation.aexecute_a2a_delegation`) -- see
crewai_prober.py's docstring for the full design rationale.

Deliberately minimal: does one basic message get a real reply? Mirrors
the other native-client layers' shallow-check scope for a fair,
apples-to-apples comparison.
"""

from __future__ import annotations

from crewai.a2a.utils.delegation import aexecute_a2a_delegation

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    try:
        result = await aexecute_a2a_delegation(
            endpoint=base_url, auth=None, timeout=30, task_description=PROBE_TEXT,
        )
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}

    status = result.get("status")
    state = status.value if hasattr(status, "value") else str(status)
    reply = result.get("result") or (result.get("error") if state == "input_required" else None)
    return {"reachable": True, "reply": reply, "error": None}
