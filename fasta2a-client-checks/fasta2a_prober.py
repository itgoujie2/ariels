"""Drives one skill through pydantic-ai's own first-party A2A client
(`fasta2a.client.A2AClient`) to a stopping point, at parity with the
golden probers' and the other native-client layers' per-skill test
depth.

`fasta2a` (github.com/pydantic/fasta2a, `pip install fasta2a`) is
maintained by the Pydantic team, primarily documented as a toolkit for
*serving* a PydanticAI agent over A2A -- but it ships a genuine,
importable client too (`fasta2a.client.A2AClient`), confirmed via direct
package inspection (not assumed). Two things make this client a
genuinely distinct data point in the native-client matrix, not just a
fifth copy of the same shape:

- **It has zero dependency on `a2a-sdk` at all** -- it's a from-scratch
  implementation of the wire protocol using its own pydantic `TypedDict`
  schema (`fasta2a/schema.py`), much like this project's own hand-rolled
  `a2a_wire.py`. Every other native client tested this session (ADK, raw
  SDK, AG2, CrewAI) is built on top of Google's `a2a-sdk` in some form;
  this is the first one that isn't.
- **It only ever speaks the v0.3 dialect** -- confirmed by reading
  `client.py` directly: the method names are hardcoded lowercase
  (`message/send`, `tasks/get`), there is no `A2A-Version` header sent at
  all, and no PascalCase v1.0 request shape exists anywhere in the
  package. Unlike the raw SDK client (auto-negotiates from the card) or
  AG2 (reads `capabilities.streaming`), fasta2a doesn't read the card at
  all before deciding how to talk -- it always sends the same, older
  request shape, full stop.

Key design decisions, confirmed live before building:

- `A2AClient` has **no card-fetching method whatsoever** -- a caller must
  discover the RPC endpoint some other way. This prober uses
  `app.a2a_wire.fetch_card()` for tolerant card discovery (same as every
  other layer), but deliberately does *not* reuse `resolve_target()` for
  picking *which* declared interface to call: that function prefers the
  most current declared dialect (our own probers' "always latest"
  policy), which can point at a v1.0-only endpoint this client can't
  actually speak to. Confirmed live against
  `insideout.luthersystems.com`: its card declares five
  `supportedInterfaces` entries, including separate v1.0 and v0.3 JSONRPC
  endpoints at *different URLs* -- `resolve_target()` picks the v1.0 one
  (first match), and fasta2a's hardcoded `message/send` against it gets a
  clean `-32601 method not found`, not because the agent can't be
  reached, but because that's the wrong endpoint for this specific
  client. `_resolve_v03_rpc_url()` does what a real fasta2a-based
  developer actually would: prefer the explicitly-v0.3-declared JSONRPC
  interface when one exists, falling back to the flat `url` (whose
  absent `protocolVersion` implies "0.3.0" by convention) or any JSONRPC
  interface otherwise. Confirmed live after the fix: the same agent's
  skill now correctly reaches its v0.3 endpoint and gets a real reply.
- The parsed response object (`send_message_response_ta.validate_json(...)`)
  is a validated pydantic `TypedDict` result -- confirmed live that
  `send_message_response_ta.dump_python(resp, by_alias=True,
  exclude_none=True)` converts it back into the *exact* raw wire-format
  dict shape (`contextId`, `kind: "task"`, kebab-case states, etc.) that
  `app.a2a_wire._normalize()` already expects for v0.3 -- confirmed
  against a real response from `2s.io`. Same reuse principle as every
  other layer: interpreting a successfully-parsed response doesn't depend
  on which client fetched it.
- `_raise_for_status()` only inspects the HTTP status code -- a
  well-formed JSON-RPC `error` envelope on a 200 response is returned
  normally as a parsed dict with an `error` key, not raised. This prober
  checks for that key explicitly, the same way `a2a_wire.send_message()`
  does for our own hand-rolled client.
- No multi-turn continuation convenience of any kind -- the caller must
  construct the next `Message` themselves with `task_id`/`context_id` set
  from the prior response, exactly like the raw-SDK layer's pattern (and
  our own `a2a_wire.py`'s).
"""

from __future__ import annotations

import uuid

import httpx
from fasta2a.client import A2AClient
from fasta2a.schema import Message, TextPart, send_message_response_ta

from app import llm
from app.a2a_wire import _normalize, explain_error, fetch_card

MAX_TURNS = 4

# Layer-specific explanation for this module's own documented finding #2
# (see module docstring): fasta2a's strict pydantic response validation
# can't parse at least two real, live agents' response shapes this
# project's own tolerant a2a_wire.py parser already handles (p0stman.com
# omits the `kind` discriminator; arcasos.com's history message is
# missing `kind`/`role`/`parts`/`messageId` entirely) -- checked before
# falling back to the generic a2a_wire.explain_error() buckets, same
# precedent as adkgo_prober.py's own _explain_bridge_error().
_LAYER_ERROR_EXPLANATIONS = [
    (
        "ValidationError",
        "fasta2a's client uses strict pydantic schema validation and rejected this agent's response "
        "shape -- real agents have been found sending spec-legal-ish shapes (a missing `kind` "
        "discriminator, a history message missing `kind`/`role`/`parts`/`messageId`) that this "
        "project's own tolerant parser handles fine but fasta2a's strict validation doesn't. A real "
        "limitation of fasta2a's own client, not necessarily a bug in the target agent.",
    ),
]


def _explain_layer_error(raw: str) -> str:
    for substring, explanation in _LAYER_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    return explain_error(raw)
CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}


def _resolve_v03_rpc_url(card: dict, base_url: str) -> str:
    """fasta2a's A2AClient only ever speaks v0.3 (see module docstring) and
    has no card-based endpoint resolution of its own at all -- reusing
    `app.a2a_wire.resolve_target()` verbatim would be wrong here, since
    that function deliberately prefers the *most current* declared
    interface (our own probers' "always latest" policy), which can be a
    v1.0-only endpoint this client can't actually talk to.

    Confirmed live against `insideout.luthersystems.com`: its card
    declares *five* `supportedInterfaces` entries, including both a v1.0
    and a v0.3 JSONRPC endpoint at different URLs. `resolve_target()`
    picks the v1.0 one (first match), and sending fasta2a's hardcoded
    `message/send` there gets a clean `-32601 method not found` --
    not a real incompatibility, just the wrong endpoint for this
    particular client. A real fasta2a-based developer would deliberately
    pick the v0.3-declared interface, since that's the only dialect their
    client understands -- this function does the same: prefer an
    explicitly-v0.3 JSONRPC interface if one is declared, then fall back
    to the flat `url` (whose absent `protocolVersion` implies "0.3.0" by
    convention anyway), then any JSONRPC interface, then `base_url`."""
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
    fetch_card(), since fasta2a's A2AClient has no card-fetching method at
    all -- a real developer would read the card some other way too."""
    try:
        card = await fetch_card(httpx_client, base_url)
    except Exception as e:
        return None, None, _explain_layer_error(f"{type(e).__name__}: {e}")
    rpc_url = _resolve_v03_rpc_url(card, base_url)
    client = A2AClient(base_url=rpc_url, http_client=httpx_client)
    return client, card, None


async def probe_skill_fasta2a(client: A2AClient, skill: dict) -> dict:
    """Returns a transcript dict shaped exactly like the golden probers'
    and the other native-client layers', so
    `app.assertions.check_transcript()` applies unchanged."""
    probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)

    turn = 0
    states: list[str] = []
    messages: list[str] = []
    turns_log: list[dict] = []
    artifacts: list[str] = []
    error: str | None = None
    final_state: str | None = None
    final_message: str | None = None
    task_id: str | None = None
    context_id: str | None = None

    next_text = probe_text
    sent_desc = probe_text

    while turn < MAX_TURNS:
        message = Message(
            role="user",
            parts=[TextPart(kind="text", text=next_text)],
            kind="message",
            message_id=str(uuid.uuid4()),
        )
        if task_id:
            message["task_id"] = task_id
        if context_id:
            message["context_id"] = context_id

        try:
            resp = await client.send_message(message)
        except Exception as e:
            error = _explain_layer_error(f"turn {turn + 1}: {type(e).__name__}: {e}")
            break
        turn += 1

        dumped = send_message_response_ta.dump_python(resp, by_alias=True, exclude_none=True)
        if "error" in dumped:
            err = dumped["error"]
            error = _explain_layer_error(f"A2A error {err.get('code')}: {err.get('message')}")
            break
        if "result" not in dumped:
            error = _explain_layer_error(
                f"fasta2a client returned neither 'result' nor 'error' (keys: {list(dumped.keys())})"
            )
            break

        normalized = await _normalize("0.3", dumped["result"], task_id, context_id)
        state = normalized["state"]
        text = normalized["agent_text"]
        task_id = normalized["task_id"] or task_id
        context_id = normalized["context_id"] or context_id
        states.append(state)
        messages.append(text)
        turns_log.append({"sent": sent_desc, "state": state, "received": text})
        artifacts.extend(normalized["artifacts"])
        final_state = state
        final_message = text

        if state in CONTINUABLE_STATES:
            if turn >= MAX_TURNS:
                error = f"did not reach a terminal state within {MAX_TURNS} turns"
                break
            followup_text = await llm.generate_followup(text, skill)
            next_text = followup_text
            sent_desc = followup_text
            continue
        elif state in POLLABLE_STATES:
            error = (
                "task is still working/submitted -- polling a task by id via "
                "get_task() was not exercised by this check (no reachable agent "
                "in the current set naturally exercises it); a real caller "
                "would need to call that separately"
            )
            break
        else:
            break

    return {
        "turn": turn, "states": states, "messages": messages, "turns_log": turns_log,
        "artifacts": artifacts, "final_state": final_state, "final_message": final_message,
        "error": error,
    }
