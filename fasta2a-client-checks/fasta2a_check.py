"""Reachability check using pydantic-ai's own first-party A2A client
(`fasta2a.client.A2AClient`) -- see fasta2a_prober.py's docstring for
the full design rationale.

Deliberately minimal: does the client resolve the card (via our own
tolerant fetch_card, since fasta2a's client has no card-fetching method
of its own) and does one basic message get a real reply? Mirrors the
other native-client layers' shallow-check scope.
"""

from __future__ import annotations

import uuid

import httpx
from fasta2a.schema import Message, TextPart, send_message_response_ta

from app.a2a_wire import _normalize
from fasta2a_prober import resolve_client

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str, httpx_client: httpx.AsyncClient) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    client, _card, resolve_error = await resolve_client(base_url, httpx_client)
    if client is None:
        return {"reachable": False, "reply": None, "error": resolve_error}

    message = Message(
        role="user", parts=[TextPart(kind="text", text=PROBE_TEXT)],
        kind="message", message_id=str(uuid.uuid4()),
    )
    try:
        resp = await client.send_message(message)
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}

    dumped = send_message_response_ta.dump_python(resp, by_alias=True, exclude_none=True)
    if "error" in dumped:
        err = dumped["error"]
        return {"reachable": False, "reply": None, "error": f"A2A error {err.get('code')}: {err.get('message')}"}
    if "result" not in dumped:
        return {"reachable": False, "reply": None, "error": "no result/error in response"}

    normalized = await _normalize("0.3", dumped["result"], None, None)
    return {"reachable": True, "reply": normalized["agent_text"], "error": None}
