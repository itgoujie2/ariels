"""Drives one skill through the raw, current, official `a2a-sdk` client
(no framework wrapper at all) to a stopping point, at parity with the
golden probers' and the ADK native-client layer's per-skill test depth.

Why "raw SDK" instead of "LangGraph native": researched and confirmed
LangGraph has no framework-specific A2A client, first-party or even
community -- unlike ADK's `RemoteA2aAgent`. The official LangGraph A2A
sample's own client demo (`test_client.py` in `a2a-samples`) is itself
just plain `a2a-sdk` usage with zero LangGraph-specific code -- and, found
along the way, that demo's own `from a2a.client import A2AClient` import
no longer works on the current `a2a-sdk` at all (`A2AClient` was replaced
by `Client`/`ClientFactory`/`create_client`); the official sample code
itself is stale relative to the SDK it demonstrates. So the honest
real-world equivalent of "a LangGraph developer calling out to A2A" is: a
developer using vanilla `a2a-sdk` directly, at whatever version `pip
install a2a-sdk` resolves to today. This also represents the realistic
calling pattern for any framework without its own dedicated client (most
of them).

Key design decisions, confirmed live before building (not guessed):
- `create_client()`/`ClientConfig` defaults to `streaming=True` -- a real,
  novel finding: this genuinely hangs/times out against real agents that
  don't handle streaming well (confirmed live against
  insideout.luthersystems.com -- times out with `streaming=True`, works
  instantly with `streaming=False`). Tested here with `streaming=False`
  explicitly (matching ADK's own default, for a fair comparison) -- the
  default-streaming-hangs finding is documented separately (see
  FINDINGS.md), not baked into this as the primary test config.
- Continuation for `input-required`: unlike ADK's `RemoteA2aAgent` (which
  wraps it as a synthetic FunctionCall), the raw client is much simpler --
  `task_id`/`context_id` are fields directly on the `Message` proto, so
  continuing is just constructing a new `Message` with them set, matching
  our own hand-rolled `a2a_wire.py`'s own pattern. Confirmed live against
  insideout.luthersystems.com: reproduces the exact same real turn-2
  backend bug found by both the golden prober and the ADK native client.
- Response interpretation reuses `app.a2a_wire._normalize()` the same way
  the ADK layer does: convert the returned Task/Message protobuf object to
  a dict (`google.protobuf.json_format.MessageToDict`) and feed it through
  the same shape-driven parser -- confirmed the resulting dict shape
  (flat, no "kind" discriminator, camelCase keys) is already one of the
  shapes `_task_from_result()`/`_extract_texts()` handle.
"""

from __future__ import annotations

import uuid

import httpx
from a2a.client import Client, ClientConfig, create_client
from a2a.types import Message, Part, Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict

from app import llm
from app.a2a_wire import _normalize, explain_error

MAX_TURNS = 4
CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}

CARD_PATHS = ["/.well-known/agent-card.json", "/.well-known/agent.json"]


async def resolve_client(base_url: str, httpx_client: httpx.AsyncClient) -> tuple[Client, str | None]:
    """Tries both known card discovery paths, same fallback the golden
    probers' fetch_card() uses. Returns (client, error) -- error is None
    on success."""
    config = ClientConfig(streaming=False, httpx_client=httpx_client)
    last_error = None
    for path in CARD_PATHS:
        try:
            client = await create_client(base_url, relative_card_path=path, client_config=config)
            return client, None
        except Exception as e:
            last_error = explain_error(f"{type(e).__name__}: {e}")
    return None, last_error


def _extract_response_dict(resp) -> dict | None:
    which = resp.WhichOneof("payload") if hasattr(resp, "WhichOneof") else None
    if which is None:
        # Fallback: check HasField directly for older/uncertain proto builds.
        if resp.HasField("task"):
            which = "task"
        elif resp.HasField("message"):
            which = "message"
        else:
            return None
    if which == "task":
        return MessageToDict(resp.task, preserving_proto_field_name=False)
    if which == "message":
        return MessageToDict(resp.message, preserving_proto_field_name=False)
    return None


async def probe_skill_raw_sdk(client: Client, skill: dict) -> dict:
    """Returns a transcript dict shaped exactly like the golden probers'
    and the ADK native-client layer's, so `app.assertions.check_transcript()`
    applies unchanged."""
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
            message_id=str(uuid.uuid4()),
            role=Role.ROLE_USER,
            task_id=task_id or "",
            context_id=context_id or "",
            parts=[Part(text=next_text)],
        )
        request = SendMessageRequest(message=message)

        raw_response = None
        turn_error = None
        try:
            async for resp in client.send_message(request):
                raw_response = _extract_response_dict(resp)
        except Exception as e:
            turn_error = explain_error(f"{type(e).__name__}: {e}")
        turn += 1

        if turn_error:
            error = turn_error
            break
        if raw_response is None:
            error = "raw a2a-sdk client returned no task/message payload"
            break

        normalized = await _normalize("1.0", raw_response, task_id, context_id)
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
