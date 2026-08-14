"""Drives one skill through Microsoft's own first-party A2A client
(`agent_framework_a2a.A2AAgent`, the `agent-framework-a2a` package) to a
stopping point, at parity with the golden probers' and the other
native-client layers' per-skill test depth.

`agent-framework` is Microsoft's official successor unifying AutoGen
and Semantic Kernel (reached v1.0 for both .NET and Python) -- a
genuinely distinct package from the `autogen-agentchat` and
`semantic-kernel` packages already ruled out earlier this session (both
confirmed to ship zero A2A support). `agent-framework-a2a` is real and
substantial: `A2AAgent`/`A2AAgentSession` wrap `a2a-sdk`'s own
`Client`/`ClientFactory`, pinning `a2a-sdk>=1.0.0,<2` -- the *current*
SDK generation, like AG2's. This is, by design, the most mature client
tested this session: `A2AAgentSession` explicitly tracks `context_id`/
`task_id`/`task_state`, and `_prepare_message_for_a2a()` correctly
threads *both* `context_id` and `task_id` back for continuation when
the prior state was `input_required` -- exactly the spec-correct
behavior CrewAI's and Agno's clients were found to lack.

Key findings, all confirmed live before/while building:

- **The single most serious finding of this whole session:
  `A2AAgent.run()`'s default, non-streaming call (`stream=False`, the
  default) silently returns EMPTY text whenever the response lands in a
  non-terminal state (`input_required`, `auth_required`, `working`,
  `submitted`) delivered as a single-shot (non-SSE) response.**
  Confirmed live against `insideout.luthersystems.com`: a real,
  substantive clarifying question came back with `resp.text == ''`,
  while `session.task_state` correctly showed `input_required` --
  reading the source confirms why: `_updates_from_task()` only surfaces
  message content from an in-progress state when `emit_intermediate=True`,
  and `run()` sets `emit_intermediate=stream` -- so the *only* way to see
  that text is to pass `stream=True`. Confirmed live: the exact same
  call with `stream=True` returns the real question text. A caller using
  Microsoft's own documented default entry point would see a legitimate,
  answerable clarifying question as if the agent had said nothing at
  all -- with no exception, no empty-string check hint, nothing. This
  prober always uses `stream=True` specifically to measure real
  capability rather than reproduce this default's data loss in its own
  numbers; the gap itself is the finding, not a bug in this prober.
- Card-based endpoint resolution: `A2AAgent(url=...)` builds only a
  *minimal* agent card assuming `url` is directly the RPC endpoint --
  no `.well-known` discovery of its own. This layer uses the same
  tolerant `app.a2a_wire.fetch_card()` + `resolve_target()` as every
  other layer (no v0.3-preference hack needed here, unlike fasta2a/
  Agno, since this client is current-generation and correctly speaks
  v1.0 -- confirmed live against `insideout.luthersystems.com`'s v1.0
  JSONRPC interface).
- Continuation confirmed live end-to-end against the same agent: turn 2,
  continuing via the session's own tracked `context_id`+`task_id`,
  reaches the *exact same* real target-side backend bug already found
  and regression-tested via five other clients this session
  (`-32603 "a2a task get: secret required"`) -- the sixth independent
  confirmation of that one real bug, this time through Microsoft's own
  official client, and further proof the client's own continuation
  logic is spec-correct (it reaches the right code path on the target's
  side to trigger the target's own pre-existing bug).
- Response interpretation reuses `app.a2a_wire._normalize()` the same
  way the raw-SDK/ADK layers do: each yielded `AgentResponseUpdate` has
  a `raw_representation` attribute carrying the actual underlying
  `a2a-sdk` protobuf `Task`/`Message` object; `MessageToDict()` on that
  (confirmed live) produces the same wire-shape dict `_normalize()`
  already expects, bypassing `AgentResponse.text` entirely (the thing
  found lossy above) for accurate measurement.
- Minor additional wrinkle on the same theme: even with `stream=True`,
  `_updates_from_task()` only emits an update for an in-progress state
  (`working`/`submitted`) when the task carries a status message at
  all -- a contentless in-progress update (no message attached, common
  for a bare "still working" ping with no text) yields *zero*
  `raw_representation` events, not even an empty one signaling the
  state. This prober reports that as "no A2A task/message payload
  returned" distinctly from the `POLLABLE_STATES` case (which fires
  when a state-bearing update *did* come through).
- Not verified live: whether `background=False` (the default) causes
  `A2AAgent` to transparently wait out a `working`/`submitted` task
  internally, as its own docstring claims ("the agent internally waits
  for the task to complete") -- no reachable agent in the current set
  both passes this client's strict v1.0 schema validation and naturally
  returns `working`/`submitted` (the one candidate,
  `rsperformance.online`, fails this client's schema entirely -- see
  next finding). Reported as a distinct, attributed non-error rather
  than guessed at.
- Real finding: `rsperformance.online`'s response (a real, live,
  hybrid-transport agent already known to bend the spec) fails this
  client's strict protobuf schema validation outright --
  `ParseError: Message type "lf.a2a.v1.SendMessageResponse" has no
  field named "id"` -- its JSON-RPC envelope includes a top-level `id`
  field alongside `result`, which this client's stricter-than-`a2a_wire.py`
  schema rejects entirely before even reaching task interpretation.
"""

from __future__ import annotations

import httpx
from agent_framework_a2a import A2AAgent, A2AAgentSession
from google.protobuf.json_format import MessageToDict

from app import llm
from app.a2a_wire import _normalize, explain_error, fetch_card, resolve_target

MAX_TURNS = 4
CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}

# Layer-specific explanation for this module's own documented finding #5
# (see module docstring): this client's strict protobuf schema
# validation rejects several real, spec-legal-ish v1.0 response
# variants outright (a `kind` discriminator present, an extra top-level
# `id` field, underscore-cased enum values) that this project's own
# tolerant parser already handles fine. Confirmed live against
# rsperformance.online: `ParseError: Message type
# "lf.a2a.v1.SendMessageResponse" has no field named "id"`. Checked
# before falling back to the generic a2a_wire.explain_error() buckets,
# same precedent as adkgo_prober.py's own _explain_bridge_error().
_LAYER_ERROR_EXPLANATIONS = [
    (
        "ParseError",
        "Microsoft Agent Framework's A2AAgent uses strict protobuf schema validation and rejected "
        "this agent's response shape -- real agents have been found sending spec-legal-ish shapes "
        "(an extra top-level \"id\" field alongside \"result\", underscore-cased enum values) that "
        "this project's own tolerant parser handles fine but this client's strict protobuf "
        "validation doesn't. A real limitation of this client's own response parsing, not "
        "necessarily a bug in the target agent.",
    ),
]


def _explain_layer_error(raw: str) -> str:
    for substring, explanation in _LAYER_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    return explain_error(raw)


async def resolve_agent(base_url: str, httpx_client: httpx.AsyncClient) -> tuple[A2AAgent | None, dict | None, str | None]:
    """Returns (agent, card, error). Card discovery is our own tolerant
    fetch_card()/resolve_target(), since A2AAgent's own url= constructor
    path has no `.well-known` discovery of its own at all."""
    try:
        card = await fetch_card(httpx_client, base_url)
    except Exception as e:
        return None, None, _explain_layer_error(f"{type(e).__name__}: {e}")
    rpc_url, _dialect = resolve_target(card, base_url)
    agent = A2AAgent(url=rpc_url, http_client=httpx_client)
    return agent, card, None


def _extract_raw(update) -> dict | None:
    raw = update.raw_representation
    if raw is None:
        return None
    return MessageToDict(raw, preserving_proto_field_name=False)


async def probe_skill_agent_framework(agent: A2AAgent, skill: dict) -> dict:
    """Returns a transcript dict shaped exactly like the golden probers'
    and the other native-client layers', so
    `app.assertions.check_transcript()` applies unchanged.

    Always calls `agent.run(..., stream=True)` -- see module docstring:
    the default non-streaming call silently drops content for any
    non-terminal state, so this is the only way to measure real
    capability rather than reproduce that gap in this prober's own
    results."""
    probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)

    turn = 0
    states: list[str] = []
    messages: list[str] = []
    turns_log: list[dict] = []
    artifacts: list[str] = []
    error: str | None = None
    final_state: str | None = None
    final_message: str | None = None
    session = A2AAgentSession()

    next_text = probe_text
    sent_desc = probe_text

    while turn < MAX_TURNS:
        last_raw = None
        try:
            stream = agent.run(next_text, session=session, stream=True)
            async for update in stream:
                raw = _extract_raw(update)
                if raw is not None:
                    last_raw = raw
        except Exception as e:
            error = _explain_layer_error(f"turn {turn + 1}: {type(e).__name__}: {e}")
            break
        turn += 1

        if last_raw is None:
            error = f"turn {turn}: agent-framework client returned no A2A task/message payload"
            break

        normalized = await _normalize("1.0", last_raw, session.task_id, session.context_id)
        state = normalized["state"]
        text = normalized["agent_text"]
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
                "task is still working/submitted -- whether A2AAgent's "
                "background=False default transparently waits this out was "
                "not exercised by this check (no reachable agent in the "
                "current set both passes this client's strict schema and "
                "naturally hits this state); a real caller would need to "
                "verify this directly"
            )
            break
        else:
            break

    return {
        "turn": turn, "states": states, "messages": messages, "turns_log": turns_log,
        "artifacts": artifacts, "final_state": final_state, "final_message": final_message,
        "error": error,
    }
