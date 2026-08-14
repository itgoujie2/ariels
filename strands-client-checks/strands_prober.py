"""Drives one skill through AWS's own first-party A2A client
(`strands.agent.a2a_agent.A2AAgent`, the `strands-agents[a2a]` extra) to
a stopping point, at parity with the golden probers' and the other
native-client layers' per-skill test depth.

Confirmed via direct package inspection: `strands-agents[a2a]` genuinely
ships `A2AAgent`, a clean, high-level client wrapper (`invoke_async`/
`stream_async`) built on top of `a2a-sdk`'s own `Client`/`ClientFactory`
(imports directly from `a2a.client`) -- structurally similar to ADK's
`RemoteA2aAgent`. Pins `a2a-sdk>=0.3.0,<0.4.0`, the same older SDK
generation as ADK/CrewAI.

Key design decisions and findings, confirmed live before/while building:

- **Real finding: `A2AAgent` provides no mechanism whatsoever to
  continue a multi-turn A2A conversation.** Read `_converters.py`'s
  `convert_input_to_message()` directly: every call builds a brand-new
  `A2AMessage` with a fresh random `message_id` and no `task_id`/
  `context_id` set -- `invoke_async(prompt, **kwargs)`'s `**kwargs` are
  explicitly documented as "ignored". This is stricter than every other
  client tested this session -- even ADK, which at least supports
  continuation for `input-required` via a synthetic `FunctionCall`.
  There is no public, documented way for a Strands developer to resume a
  task at all. This prober therefore sends exactly **one turn per
  skill** -- `input-required`/`auth-required` is treated as a legitimate
  blocked stopping point (same treatment `assertions.py` already gives
  those states for any layer), not a failed continuation attempt, since
  continuing genuinely isn't something this client's public API can do.
- **Real finding: `_get_a2a_client()` forces `streaming=True` in its own
  `ClientConfig`** unconditionally, even overriding a caller-supplied
  `client_config.streaming=False` (`dataclasses.replace(self._client_config,
  streaming=True)` in the source) -- a real caller has no supported way
  to opt out of that preference at all, unlike the raw-SDK layer
  (defaults to streaming but is overridable) or AG2 (reads the card).
  In practice the underlying `a2a-sdk` `ClientFactory` still appears to
  fall back to plain `message/send` against a card that doesn't itself
  declare `capabilities.streaming` (confirmed live via a mocked card),
  so this is a preference override more than a guaranteed behavior
  change -- noted as a real, confirmed source-level finding, not
  verified to cause a hang the way the raw-SDK layer's equivalent
  default did.
- **Real finding, more serious: `convert_response_to_agent_result()`
  drops the agent's reply text entirely for a common, real response
  shape.** Confirmed via a mocked card+response (no live agent needed to
  demonstrate this one): when a completed task arrives as a plain
  `(Task, None)` tuple -- no separate `TaskStatusUpdateEvent`/
  `TaskArtifactUpdateEvent`, i.e. exactly how a non-streaming
  `message/send` response naturally looks -- the converter's tuple
  branch only ever reads reply text from `update_event.status.message`
  (`TaskStatusUpdateEvent`) or `update_event.artifact`
  (`TaskArtifactUpdateEvent`); with `update_event` being `None`, neither
  branch runs, and it *never* falls back to reading `task.status.message`
  directly -- only `task.artifacts`. A real agent that answers via
  `status.message` and never populates `artifacts` (this project has
  already found several: `agent.co-legal.be`, `MolTrust`, `InsideOut`'s
  own completions) would hand a Strands-based caller back a
  **completely empty reply** through the public, documented
  `invoke_async`/`__call__` API, despite the agent answering correctly.
- **Real finding: the high-level `AgentResult`/`stop_reason` abstraction
  silently loses the real A2A task state for a common response shape.**
  `_extract_task_state()` (`_converters.py`) only reads state from a
  separate `TaskStatusUpdateEvent` -- but confirmed live against `2s.io`:
  a single-shot, fully-`completed` response arrives as `(Task, None)`
  (no separate update event), so `_extract_task_state()` returns `None`
  and `stop_reason` defaults to a generic `"end_turn"` with an **empty**
  `state` dict. The real `Task.status.state` sits right there on the
  discarded first tuple element, unused. Confirmed by reading the source
  that this collapses `completed`/`failed`/`canceled`/`rejected` into
  the exact same signal whenever no separate status-update event
  accompanies the response -- a caller using only the public
  `AgentResult` can't tell a clean success from a failure in that shape.
  This prober does **not** rely on `AgentResult`/`stop_reason` for that
  reason -- it uses the lower-level `stream_async()` events directly,
  extracts the raw underlying `Task`/`Message` object from the last
  `a2a_stream` event (a2a-sdk 0.3.x's `Task`/`Message` are pydantic
  models here, not protobuf -- confirmed live, `model_dump(mode="json",
  by_alias=True, exclude_none=True)` produces the exact camelCase wire
  shape `app.a2a_wire._normalize()` already expects), and reads the real
  state from `task.status.state` itself. Same reuse principle as every
  other layer: interpreting a successfully-returned response doesn't
  depend on which client fetched it -- but this layer specifically
  bypasses Strands' own convenience conversion to get an accurate
  measurement, since that conversion is the thing found to lose data.
"""

from __future__ import annotations

from strands.agent.a2a_agent import A2AAgent

from app import llm
from app.a2a_wire import _normalize, explain_error

CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}

# Layer-specific explanation, same finding already confirmed for
# crewai-client-checks and langchain4j-client-checks: Strands' own
# A2AAgent is constructed with the raw base_url (see run_parity.py's
# `A2AAgent(endpoint=base_url, ...)`), so it does its own independent
# card fetch when a skill is actually probed -- separate from, and more
# limited than, this project's own tolerant a2a_wire.fetch_card()
# (which already succeeded by this point, that's how the report's
# skill list was enumerated). Confirmed live: A2AClientHTTPError's own
# message embeds "Failed to fetch agent card from <url>" whenever this
# fires. Checked before falling back to the generic
# a2a_wire.explain_error() buckets, same precedent as every other
# layer's own _explain_layer_error().
_LAYER_ERROR_EXPLANATIONS = [
    (
        "Failed to fetch agent card",
        "Strands' A2AAgent client only tries the current A2A spec's well-known path "
        "(/.well-known/agent-card.json) -- it has no fallback to the older, legacy path "
        "(/.well-known/agent.json) that some real agents (like p0stman.com) still use exclusively. "
        "This project's own broader discovery already found this agent's card successfully (that's "
        "how its skills were enumerated for this report), which proves the card exists -- this is a "
        "limitation of Strands' own client, not a genuinely undiscoverable agent.",
    ),
]


def _explain_layer_error(raw: str) -> str:
    for substring, explanation in _LAYER_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    return explain_error(raw)


def _extract_raw_task_or_message(a2a_event) -> dict | None:
    """`a2a_event` is whatever `A2AStreamEvent` wraps: either a bare
    `Message` or a `(Task, update_event | None)` tuple. Always dumps the
    `Task` itself (not the possibly-partial update_event) so the real
    `status.state` is never lost, unlike Strands' own
    `convert_response_to_agent_result()`."""
    if isinstance(a2a_event, tuple) and len(a2a_event) == 2:
        task, _update = a2a_event
        return task.model_dump(mode="json", by_alias=True, exclude_none=True)
    if a2a_event is not None and hasattr(a2a_event, "model_dump"):
        return a2a_event.model_dump(mode="json", by_alias=True, exclude_none=True)
    return None


async def probe_skill_strands(agent: A2AAgent, skill: dict) -> dict:
    """Returns a transcript dict shaped exactly like the golden probers'
    and the other native-client layers', so
    `app.assertions.check_transcript()` applies unchanged.

    Always exactly one turn -- see module docstring: `A2AAgent` has no
    continuation mechanism at all, so `input-required`/`auth-required`
    is reported as a legitimate blocked terminal state on turn 1, not a
    failed follow-up attempt."""
    probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)

    error: str | None = None
    last_raw_event = None
    try:
        async for event in agent.stream_async(probe_text):
            if event.get("type") == "a2a_stream":
                last_raw_event = event["event"]
    except Exception as e:
        error = _explain_layer_error(f"{type(e).__name__}: {e}")

    if error:
        return {
            "turn": 0, "states": [], "messages": [], "turns_log": [],
            "artifacts": [], "final_state": None, "final_message": None, "error": error,
        }

    raw = _extract_raw_task_or_message(last_raw_event)
    if raw is None:
        return {
            "turn": 1, "states": [], "messages": [], "turns_log": [],
            "artifacts": [], "final_state": None, "final_message": None,
            "error": "strands A2AAgent produced no A2A task/message event",
        }

    normalized = await _normalize("1.0", raw, None, None)
    state = normalized["state"]
    text = normalized["agent_text"]

    turn_error = None
    if state in POLLABLE_STATES:
        turn_error = (
            "task is still working/submitted -- polling a task by id was not "
            "exercised by this check (A2AAgent's public API has no polling "
            "mechanism at all, matching its lack of any continuation support); "
            "a real caller would need to call the underlying a2a-sdk client directly"
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
