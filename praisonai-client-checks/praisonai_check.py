"""Reachability check using PraisonAI's own A2A client
(`praisonai.endpoints.providers.a2a.A2AProvider`) -- see
praisonai_prober.py's docstring for the full design rationale.

Deliberately minimal: does one basic message get a real reply? Mirrors
the other native-client layers' shallow-check scope.
"""

from __future__ import annotations

import asyncio

from praisonai.endpoints.providers.a2a import A2AProvider

from app.a2a_wire import _normalize

PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}."""
    provider = A2AProvider(base_url=base_url, timeout=30)
    try:
        result = await asyncio.to_thread(
            provider.invoke, name="__probe__", input_data={"message": PROBE_TEXT}
        )
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}

    if not result.ok:
        return {"reachable": False, "reply": None, "error": str(result.error)}

    try:
        normalized = await _normalize("1.0", result.data, None, None)
    except Exception as e:
        return {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}
    return {"reachable": True, "reply": normalized["agent_text"], "error": None}
