"""Reachability check using Microsoft's own first-party A2A client
(`agent_framework_a2a.A2AAgent`) -- see agent_framework_prober.py's
docstring for the full design rationale.

Deliberately minimal: does one basic message get a real reply? Mirrors
the other native-client layers' shallow-check scope. Uses
`stream=True` for the same reason the full-parity prober does -- the
non-streaming default silently drops content for non-terminal states.
"""

from __future__ import annotations

import httpx

from app.a2a_wire import _normalize
from agent_framework_prober import _extract_raw, resolve_agent

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str, httpx_client: httpx.AsyncClient) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    agent, _card, resolve_error = await resolve_agent(base_url, httpx_client)
    if agent is None:
        return {"reachable": False, "reply": None, "error": resolve_error}

    last_raw = None
    try:
        stream = agent.run(PROBE_TEXT, stream=True)
        async for update in stream:
            raw = _extract_raw(update)
            if raw is not None:
                last_raw = raw
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}

    if last_raw is None:
        return {"reachable": False, "reply": None, "error": "no A2A task/message payload returned"}

    normalized = await _normalize("1.0", last_raw, None, None)
    return {"reachable": True, "reply": normalized["agent_text"], "error": None}
