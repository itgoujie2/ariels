"""Drives one skill through Google ADK's real, official A2A client
(`RemoteA2aAgent`) to a stopping point, at parity with the golden probers'
per-skill test depth (multi-turn continuation, same assertion checks) --
not just the shallow reachability check in `adk_check.py`.

Key design decision: RemoteA2aAgent's own request-sending and turn-
continuation machinery is what's under test here (that's the whole point
of this layer -- see PLAN.md §11a) -- but *interpreting* whatever raw
response it successfully gets back reuses our own already-tolerant
`app.a2a_wire._normalize()`/`_task_from_result()`/`_extract_texts()`
rather than reinventing state/content parsing against ADK's Event
abstraction. This is legitimate, not circumventing the test: once
RemoteA2aAgent has successfully round-tripped a raw response, its meaning
doesn't depend on which framework fetched it. Confirmed live
(`event.custom_metadata["a2a:response"]` carries the exact raw task/
message dict RemoteA2aAgent received, verified against p0stman.com) that
this metadata is populated on success.

Continuation mechanics, confirmed by reading `remote_a2a_agent.py` and
verified live against insideout.luthersystems.com (a real agent that
naturally hits input-required):
  - `input-required` is surfaced as a synthetic ("mock") FunctionCall
    named `mock_function_call_for_required_user_input`, with the agent's
    prompt text as `args["input_required"]`. Continuing means sending a
    new turn whose message is a matching `FunctionResponse` with
    `response={"result": <our reply text>}` -- ADK detects this pattern
    from session history and reconstructs the correct A2A continuation
    request (same task/context id) automatically.
  - `working`/`submitted` get NO mock function call under RemoteA2aAgent's
    default client config (`polling=False`, `streaming=False` --
    confirmed by reading `_ensure_httpx_client`) -- there is no continuation
    handle exposed at all. A real caller using the default, out-of-the-box
    configuration would see the task as stuck, with no way to check back
    on it. That's reported as a real, distinct finding, not worked around
    (working around it would mean testing a non-default configuration most
    real callers wouldn't actually use).
"""

from __future__ import annotations

import httpx
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from app import llm
from app.a2a_wire import _normalize, explain_error

MAX_TURNS = 4
CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}

MOCK_FC_INPUT = "mock_function_call_for_required_user_input"
MOCK_FC_AUTH = "mock_function_call_for_required_user_auth"


def _find_function_call(event, fc_id: str):
    if not event.content or not event.content.parts:
        return None
    for part in event.content.parts:
        if part.function_call and part.function_call.id == fc_id:
            return part.function_call
    return None


async def probe_skill_native(agent_card_url: str, skill: dict, httpx_client: httpx.AsyncClient) -> dict:
    """Returns a transcript dict shaped exactly like the golden probers'
    (turn, states, messages, turns_log, artifacts, final_state,
    final_message, error) so `app.assertions.check_transcript()` can be
    reused unchanged."""
    probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)

    turn = 0
    states: list[str] = []
    messages: list[str] = []
    turns_log: list[dict] = []
    artifacts: list[str] = []
    error: str | None = None
    final_state: str | None = None
    final_message: str | None = None

    try:
        agent = RemoteA2aAgent(name="native_probe", agent_card=agent_card_url, httpx_client=httpx_client)
        runner = InMemoryRunner(agent=agent)
        session = await runner.session_service.create_session(app_name=runner.app_name, user_id="probe")
    except Exception as e:
        return {
            "turn": 0, "states": [], "messages": [], "turns_log": [], "artifacts": [],
            "final_state": None, "final_message": None, "error": explain_error(f"{type(e).__name__}: {e}"),
        }

    next_message = genai_types.Content(role="user", parts=[genai_types.Part(text=probe_text)])
    sent_desc = probe_text

    while turn < MAX_TURNS:
        last_event = None
        turn_error = None
        try:
            async for event in runner.run_async(user_id="probe", session_id=session.id, new_message=next_message):
                last_event = event
                if event.error_message:
                    turn_error = explain_error(event.error_message)
        except Exception as e:
            turn_error = explain_error(f"{type(e).__name__}: {e}")
        turn += 1

        if turn_error:
            error = turn_error
            break
        if last_event is None:
            error = "native client returned no event"
            break

        raw_response = (last_event.custom_metadata or {}).get("a2a:response")
        if raw_response is None:
            error = "native client returned no a2a:response metadata to interpret"
            break

        normalized = await _normalize("1.0", raw_response, None, None)
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
            if not last_event.long_running_tool_ids:
                error = "input-required state but native client exposed no continuation handle"
                break
            fc_id = next(iter(last_event.long_running_tool_ids))
            fc = _find_function_call(last_event, fc_id)
            if fc is None or fc.name not in (MOCK_FC_INPUT, MOCK_FC_AUTH):
                error = "could not find matching mock function_call to continue"
                break
            followup_text = await llm.generate_followup(text, skill)
            next_message = genai_types.Content(
                role="user",
                parts=[genai_types.Part(function_response=genai_types.FunctionResponse(
                    id=fc_id, name=fc.name, response={"result": followup_text},
                ))],
            )
            sent_desc = followup_text
            continue
        elif state in POLLABLE_STATES:
            error = (
                "task is still working/submitted, and RemoteA2aAgent's default "
                "configuration (polling=False) exposes no way to check back on it "
                "-- a real caller using the out-of-the-box client would see this "
                "agent as permanently unresponsive"
            )
            break
        else:
            break

    return {
        "turn": turn, "states": states, "messages": messages, "turns_log": turns_log,
        "artifacts": artifacts, "final_state": final_state, "final_message": final_message,
        "error": error,
    }
