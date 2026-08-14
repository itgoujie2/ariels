"""Reachability check using the raw, current, official `a2a-sdk` client
(no framework wrapper) -- see raw_sdk_prober.py's docstring for the full
design rationale (why "raw SDK" replaces "LangGraph native").

Deliberately minimal: does the client resolve the card, and does sending
one basic message get a real reply? Mirrors
`native-client-checks/adk_check.py`'s scope for a fair, apples-to-apples
comparison between the two native-client layers.
"""

from __future__ import annotations

import uuid

import httpx
from a2a.types import Message, Part, Role, SendMessageRequest

from raw_sdk_prober import _extract_response_dict, resolve_client
from app.a2a_wire import _normalize

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str, httpx_client: httpx.AsyncClient) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    client, resolve_error = await resolve_client(base_url, httpx_client)
    if client is None:
        return {"reachable": False, "reply": None, "error": resolve_error}

    message = Message(
        message_id=str(uuid.uuid4()), role=Role.ROLE_USER, parts=[Part(text=PROBE_TEXT)],
    )
    request = SendMessageRequest(message=message)
    try:
        raw_response = None
        async for resp in client.send_message(request):
            raw_response = _extract_response_dict(resp)
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}

    if raw_response is None:
        return {"reachable": False, "reply": None, "error": "no task/message payload returned"}

    normalized = await _normalize("1.0", raw_response, None, None)
    return {"reachable": True, "reply": normalized["agent_text"], "error": None}
