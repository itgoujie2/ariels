"""Reachability check using Mastra's own first-party A2A client
(`@mastra/core/a2a`'s `A2AAgent`, via the Node bridge process) -- see
mastra_prober.py's docstring for the full design rationale.

Deliberately minimal: does one basic message get a real reply? Mirrors
the other native-client layers' shallow-check scope.
"""

from __future__ import annotations

import uuid

from app.a2a_wire import _normalize
from mastra_prober import resolve_bridge

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    bridge, _card, resolve_error = await resolve_bridge(base_url)
    if bridge is None:
        return {"reachable": False, "reply": None, "error": resolve_error}

    try:
        result = await bridge.call({"cmd": "generate", "runId": str(uuid.uuid4()), "text": PROBE_TEXT})
    except Exception as e:
        await bridge.close()
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}
    await bridge.close()

    if not result.get("ok"):
        return {"reachable": False, "reply": None, "error": result.get("error", "unknown bridge error")}

    raw = result.get("task") or result.get("message")
    if raw is None:
        return {"reachable": False, "reply": None, "error": "mastra bridge returned neither task nor message"}

    normalized = await _normalize("1.0", raw, None, None)
    return {"reachable": True, "reply": normalized["agent_text"], "error": None}
