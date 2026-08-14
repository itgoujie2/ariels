"""Reachability check using LangChain4j's own first-party A2A client
(via the one-shot Java bridge process) -- see langchain4j_prober.py's
docstring for the full design rationale.

Deliberately minimal: does one basic message get a real reply? Mirrors
the other native-client layers' shallow-check scope.
"""

from __future__ import annotations

from app.a2a_wire import _normalize
from langchain4j_prober import _call_bridge, resolve_bridge

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    _url, resolve_error = await resolve_bridge(base_url)
    if resolve_error:
        return {"reachable": False, "reply": None, "error": resolve_error}

    result = await _call_bridge(base_url, PROBE_TEXT, None, None)
    if not result.get("ok"):
        return {"reachable": False, "reply": None, "error": result.get("error", "unknown bridge error")}

    raw = result.get("task")
    if raw is None:
        return {"reachable": False, "reply": None, "error": "langchain4j bridge returned no task"}

    normalized = await _normalize("1.0", raw, None, None)
    return {"reachable": True, "reply": normalized["agent_text"], "error": None}
