"""Reachability check using Agno's own first-party A2A client
(`agno.client.a2a.client.A2AClient`) -- see agno_prober.py's docstring
for the full design rationale.

Deliberately minimal: does one basic message get a real reply? Mirrors
the other native-client layers' shallow-check scope.
"""

from __future__ import annotations

import httpx

from app.a2a_wire import _normalize
from agno_prober import _send_raw, resolve_client

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str, httpx_client: httpx.AsyncClient) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    client, _card, resolve_error = await resolve_client(base_url, httpx_client)
    if client is None:
        return {"reachable": False, "reply": None, "error": resolve_error}

    try:
        body = await _send_raw(client, httpx_client, PROBE_TEXT, None)
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}

    if "error" in body:
        err = body["error"]
        return {"reachable": False, "reply": None, "error": f"A2A error {err.get('code')}: {err.get('message')}"}
    if "result" not in body:
        return {"reachable": False, "reply": None, "error": "no result/error in response"}

    normalized = await _normalize("1.0", body["result"], None, None)
    return {"reachable": True, "reply": normalized["agent_text"], "error": None}
