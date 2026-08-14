"""Drives one skill through Agno's own first-party A2A client
(`agno.client.a2a.client.A2AClient`, the `agno[a2a]` extra) to a
stopping point, at parity with the golden probers' and the other
native-client layers' per-skill test depth.

Confirmed via direct package inspection: `agno[a2a]` genuinely ships a
hand-rolled A2A client (`agno/client/a2a/client.py`) -- not built on
`a2a-sdk`'s `Client`/`ClientFactory` despite the extra requiring
`a2a-sdk<1.0,>=0.3.0` as a dependency (used only by Agno's own *server*
side, `agno/os/interfaces/a2a/`). Before settling on Agno, confirmed via
direct package inspection that Letta and Camel-AI ship no A2A support
at all.

Key design decisions and findings, confirmed live before/while building:

- **Real finding: the client's default `protocol="rest"` constructs a
  URL (`{base_url}/v1/message:send`) that gets a clean 404 against any
  real, generic third-party A2A agent** -- confirmed live against
  `2s.io`. That REST-style path is specific to Agno's own server
  (`agno/os/interfaces/a2a/router.py` serves exactly that route); a real
  developer must already know to explicitly pass `protocol="json-rpc"`
  (which posts the same JSON-RPC-shaped body to the plain base URL
  instead) to talk to any agent that isn't itself an Agno server. This
  prober always uses `protocol="json-rpc"`, the only mode that actually
  interoperates with third-party agents.
- **Real finding: `get_agent_card()`/`aget_agent_card()` are broken
  specifically in `protocol="json-rpc"` mode.** `_get_endpoint()`
  ignores whatever path is passed and returns bare `base_url` whenever
  `protocol == "json-rpc"` -- so calling `aget_agent_card()` in the one
  protocol mode that actually works against real agents re-requests the
  RPC endpoint itself instead of `/.well-known/agent-card.json`, and
  gets back a JSON-RPC error body, not a card. There is effectively no
  working card-based endpoint discovery in this client at all for
  non-Agno agents. This prober uses `app.a2a_wire.fetch_card()` +
  `resolve_target()` for tolerant discovery instead (same as every other
  layer), then points `A2AClient` directly at the resolved RPC URL.
- **Real finding: `send_message()`'s public signature has no `task_id`
  parameter at all, only `context_id`.** Confirmed live against
  `insideout.luthersystems.com`: continuing with only `context_id` set
  doesn't error -- it silently starts a **new** task each time (a
  different `task_id` comes back on every call), the agent re-asking its
  generic opening question rather than resuming the real conversation.
  The exact same footgun class already found in CrewAI's client, but
  here there is no escape hatch at all -- there is no way to pass
  `task_id` even if a caller wanted to. This prober therefore sends
  exactly **one turn per skill**, treating `input-required`/
  `auth-required` as a legitimate blocked stopping point rather than a
  failed continuation attempt, since continuing genuinely isn't
  something this client's public API can do.
- **Real finding: `_parse_task_result()` never reads artifact text into
  `content` -- only artifact metadata** (`artifact_id`/`name`/
  `description`/`mime_type`/`uri`), never the actual `parts` of each
  artifact. Confirmed live against `2s.io` (which answers entirely via
  artifacts): `status="completed"` with a real, non-empty `artifacts`
  list, but `content` comes back as an **empty string** -- the same
  class of content-extraction gap this project's own `a2a_wire.py` was
  fixed for early this session, now found in a fifth third-party
  client. This prober does not rely on `TaskResult.content` for that
  reason -- it reuses the client's own private `_build_message_request()`
  (exactly what `send_message()` calls internally, so the request sent
  is byte-identical to a real caller's) but posts it directly and
  interprets the raw JSON-RPC response through
  `app.a2a_wire._normalize()`, which already reads both artifacts and
  history/status.message correctly.
"""

from __future__ import annotations

import httpx
from agno.client.a2a.client import A2AClient

from app import llm
from app.a2a_wire import _normalize, explain_error, fetch_card

CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}

# Layer-specific explanation for this module's own documented finding #1
# (see module docstring): Agno's client only reliably interoperates with
# third-party agents in protocol="json-rpc" mode -- its default "rest"
# mode posts to an Agno-server-specific path and gets a clean 404
# against any real, non-Agno agent. This prober always uses "json-rpc"
# to avoid that, so a 404 surfacing here more likely means the resolved
# RPC endpoint itself is wrong for this agent -- still worth surfacing
# this known client quirk as context. Checked before falling back to
# the generic a2a_wire.explain_error() buckets, same precedent as
# adkgo_prober.py's own _explain_bridge_error().
_LAYER_ERROR_EXPLANATIONS = [
    (
        "404",
        "Agno's client only reliably interoperates with third-party agents when explicitly "
        "configured for JSON-RPC mode -- its default \"rest\" mode posts to an Agno-server-specific "
        "endpoint path and returns a clean 404 against any other agent. This check always uses "
        "JSON-RPC mode to avoid that, so this 404 more likely means the resolved RPC endpoint "
        "itself doesn't accept this request -- but it's the same class of gap as Agno's own known "
        "REST-mode default quirk.",
    ),
]


def _explain_layer_error(raw: str) -> str:
    for substring, explanation in _LAYER_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    return explain_error(raw)


def _resolve_v03_rpc_url(card: dict, base_url: str) -> str:
    """Agno's client is v0.3-only too (lowercase `message/send`, no
    `A2A-Version` header anywhere -- confirmed by reading the source),
    the same situation the fasta2a layer already found and fixed for
    itself. Reusing `app.a2a_wire.resolve_target()` verbatim would be
    wrong here for the identical reason: it prefers the most current
    declared dialect, which can be a v1.0-only endpoint this client
    can't speak to. Confirmed live against `insideout.luthersystems.com`
    (whose card declares both a v1.0 and a v0.3 JSONRPC interface at
    different URLs): `resolve_target()` picks the v1.0 one, and Agno's
    `message/send` against it returns a clean `-32601 method not found`."""
    for iface in card.get("supportedInterfaces", []):
        binding = (iface.get("protocolBinding") or "").upper()
        version = (iface.get("protocolVersion") or "")[:3]
        if binding in ("JSONRPC", "JSON-RPC", "JSON_RPC") and version == "0.3":
            return iface.get("url", card.get("url", base_url))
    if "url" in card:
        return card["url"]
    for iface in card.get("supportedInterfaces", []):
        binding = (iface.get("protocolBinding") or "").upper()
        if binding in ("JSONRPC", "JSON-RPC", "JSON_RPC"):
            return iface.get("url", base_url)
    return base_url


async def resolve_client(base_url: str, httpx_client: httpx.AsyncClient) -> tuple[A2AClient | None, dict | None, str | None]:
    """Returns (client, card, error). Card discovery is our own tolerant
    fetch_card()/`_resolve_v03_rpc_url()`, since Agno's own card
    discovery is broken in the one protocol mode ("json-rpc") that
    actually works against non-Agno agents (see module docstring)."""
    try:
        card = await fetch_card(httpx_client, base_url)
    except Exception as e:
        return None, None, _explain_layer_error(f"{type(e).__name__}: {e}")
    rpc_url = _resolve_v03_rpc_url(card, base_url)
    client = A2AClient(base_url=rpc_url, timeout=30, protocol="json-rpc")
    return client, card, None


async def _send_raw(client: A2AClient, httpx_client: httpx.AsyncClient, message: str, context_id: str | None) -> dict:
    """Builds the exact request `A2AClient.send_message()` would build
    and send (reusing its own private builder, so this is byte-identical
    to a real call) but returns the raw parsed JSON-RPC body instead of
    Agno's own lossy `TaskResult` conversion."""
    request_body = client._build_message_request(message=message, context_id=context_id, stream=False)
    resp = await httpx_client.post(client._get_endpoint(path="/v1/message:send"), json=request_body, timeout=30)
    resp.raise_for_status()
    return resp.json()


async def probe_skill_agno(client: A2AClient, httpx_client: httpx.AsyncClient, skill: dict) -> dict:
    """Returns a transcript dict shaped exactly like the golden probers'
    and the other native-client layers', so
    `app.assertions.check_transcript()` applies unchanged.

    Always exactly one turn -- see module docstring: `A2AClient` has no
    way to pass a `task_id` back for continuation at all, so
    `input-required`/`auth-required` is reported as a legitimate blocked
    terminal state on turn 1, not a failed follow-up."""
    probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)

    try:
        body = await _send_raw(client, httpx_client, probe_text, None)
    except Exception as e:
        return {
            "turn": 0, "states": [], "messages": [], "turns_log": [],
            "artifacts": [], "final_state": None, "final_message": None,
            "error": _explain_layer_error(f"turn 1: {type(e).__name__}: {e}"),
        }

    if "error" in body:
        err = body["error"]
        return {
            "turn": 1, "states": [], "messages": [], "turns_log": [],
            "artifacts": [], "final_state": None, "final_message": None,
            "error": _explain_layer_error(f"A2A error {err.get('code')}: {err.get('message')}"),
        }
    if "result" not in body:
        return {
            "turn": 1, "states": [], "messages": [], "turns_log": [],
            "artifacts": [], "final_state": None, "final_message": None,
            "error": _explain_layer_error(
                f"agno client returned neither 'result' nor 'error' (keys: {list(body.keys())})"
            ),
        }

    normalized = await _normalize("1.0", body["result"], None, None)
    state = normalized["state"]
    text = normalized["agent_text"]

    turn_error = None
    if state in POLLABLE_STATES:
        turn_error = (
            "task is still working/submitted -- polling a task by id was not "
            "exercised by this check (A2AClient has no tasks/get method at all); "
            "a real caller would need to build that separately"
        )

    return {
        "turn": 1,
        "states": [state],
        "messages": [text],
        "turns_log": [{"sent": probe_text, "state": state, "received": text}],
        "artifacts": normalized["artifacts"],
        "final_state": state,
        "final_message": text,
        "error": turn_error,
    }
