"""Generic A2A prober's multi-turn state machine, built as a Google ADK
custom agent.

Same protocol conformance logic as
`reference-agents/ours/langgraph-prober/app/prober.py` -- identical
TurnState fields, identical send_initial/send_followup/poll_task node
logic, identical MAX_TURNS/CONTINUABLE_STATES/POLLABLE_STATES/routing
rules. Only the orchestration engine differs: a LangGraph StateGraph
there, an ADK BaseAgent custom control loop here -- ADK's own documented
idiom for orchestration that doesn't fit the declarative
Sequential/Loop/Parallel workflow agents (see PLAN.md §11).

Deliberately does NOT use ADK's own first-party A2A client,
`google.adk.agents.remote_a2a_agent.RemoteA2aAgent`, even though it exists
and is officially supported: RemoteA2aAgent is a thin wrapper around
a2a-sdk's own `Client`/`ClientFactory` -- exactly the dialect-abstracting
SDK behavior this whole project exists to see past (see CLAUDE.md's note
on why `a2a_wire.py` hand-rolls the wire protocol instead of depending on
a2a-sdk at all). Using it here would silently work around the real
interoperability bugs this tool is built to surface.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event

from app import llm
from app.a2a_wire import explain_error, get_task, send_message

MAX_TURNS = 4
# Only input-required is resolvable by continuing the chat. auth-required
# means the spec's own out-of-band auth flow is needed -- a text reply
# can't satisfy it, so the prober must stop there, not try to talk past it.
CONTINUABLE_STATES = {"input-required"}
# working/submitted mean the task is still processing on the agent's own
# side -- resolvable by polling tasks/get, not by sending it a new chat
# message. Confirmed live (rsperformance.online, see the LangGraph
# prober's version): a real agent replies synchronously with `working`
# and simply never proceeds via message/send alone.
POLLABLE_STATES = {"working", "submitted"}
POLL_DELAY_SECONDS = 1.5


@dataclass
class TurnState:
    skill: dict  # {"id", "name", "description", "examples": [...]}
    httpx_client: httpx.AsyncClient
    rpc_url: str = ""
    dialect: str = "0.3"
    extra_headers: dict = field(default_factory=dict)
    task_id: str | None = None
    context_id: str | None = None
    turn: int = 0
    states: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    turns_log: list[dict] = field(default_factory=list)  # [{"sent": ..., "state": ..., "received": ...}]
    artifacts: list[str] = field(default_factory=list)
    final_state: str | None = None
    final_message: str | None = None
    error: str | None = None
    elapsed_ms: float = 0.0


async def send_initial(state: TurnState) -> dict:
    probe_text = (
        state.skill["examples"][0]
        if state.skill.get("examples")
        else await llm.generate_probe(state.skill)
    )
    start = time.monotonic()
    try:
        result = await send_message(
            state.httpx_client,
            state.rpc_url,
            state.dialect,
            probe_text,
            extra_headers=state.extra_headers,
        )
    except Exception as e:
        # Loud failures (e.g. UnrecognizedResponseShapeError, HTTP errors)
        # end this skill's probe with a clear, attributed error rather than
        # crashing the whole multi-skill run.
        return {"error": explain_error(f"{type(e).__name__}: {e}")}
    return {
        "task_id": result["task_id"],
        "context_id": result["context_id"],
        "turn": state.turn + 1,
        "states": state.states + [result["state"]],
        "messages": state.messages + [result["agent_text"]],
        "turns_log": state.turns_log
        + [{"sent": probe_text, "state": result["state"], "received": result["agent_text"]}],
        "artifacts": state.artifacts + result["artifacts"],
        "final_state": result["state"],
        "final_message": result["agent_text"],
        "elapsed_ms": (time.monotonic() - start) * 1000,
    }


async def send_followup(state: TurnState) -> dict:
    if state.turn >= MAX_TURNS:
        return {"error": f"did not reach a terminal state within {MAX_TURNS} turns"}

    followup_text = await llm.generate_followup(state.messages[-1], state.skill)
    start = time.monotonic()
    try:
        result = await send_message(
            state.httpx_client,
            state.rpc_url,
            state.dialect,
            followup_text,
            task_id=state.task_id,
            context_id=state.context_id,
            extra_headers=state.extra_headers,
        )
    except Exception as e:
        return {"error": explain_error(f"{type(e).__name__}: {e}")}
    return {
        "turn": state.turn + 1,
        "states": state.states + [result["state"]],
        "messages": state.messages + [result["agent_text"]],
        "turns_log": state.turns_log
        + [{"sent": followup_text, "state": result["state"], "received": result["agent_text"]}],
        "artifacts": state.artifacts + result["artifacts"],
        "final_state": result["state"],
        "final_message": result["agent_text"],
        "elapsed_ms": state.elapsed_ms + (time.monotonic() - start) * 1000,
    }


async def poll_task(state: TurnState) -> dict:
    if state.turn >= MAX_TURNS:
        return {"error": f"did not reach a terminal state within {MAX_TURNS} turns"}
    if not state.task_id:
        # Shouldn't happen -- working/submitted only ever comes from a Task
        # object, which always carries an id -- but fail cleanly rather
        # than polling with a None id if some agent proves otherwise.
        return {"error": "cannot poll tasks/get: no task_id was returned"}

    await asyncio.sleep(POLL_DELAY_SECONDS)
    start = time.monotonic()
    try:
        result = await get_task(
            state.httpx_client,
            state.rpc_url,
            state.dialect,
            state.task_id,
            context_id=state.context_id,
            extra_headers=state.extra_headers,
        )
    except Exception as e:
        return {"error": explain_error(f"{type(e).__name__}: {e}")}
    return {
        "turn": state.turn + 1,
        "states": state.states + [result["state"]],
        "messages": state.messages + [result["agent_text"]],
        "turns_log": state.turns_log
        + [{"sent": "(polled tasks/get)", "state": result["state"], "received": result["agent_text"]}],
        "artifacts": state.artifacts + result["artifacts"],
        "final_state": result["state"],
        "final_message": result["agent_text"],
        "elapsed_ms": state.elapsed_ms + (time.monotonic() - start) * 1000,
    }


def route_after_send(state: TurnState) -> str:
    """Shared routing logic after send_initial, send_followup, or poll_task
    -- which step comes next only depends on the state we just landed in,
    not which step we came from."""
    if state.error:
        return "done"
    if state.final_state in POLLABLE_STATES and state.turn < MAX_TURNS:
        return "poll"
    if state.final_state in CONTINUABLE_STATES and state.turn < MAX_TURNS:
        return "continue"
    return "done"


def _apply(state: TurnState, updates: dict) -> TurnState:
    """Merge a node function's returned updates into a TurnState -- the
    ADK-loop equivalent of what LangGraph's StateGraph does automatically
    between nodes."""
    for k, v in updates.items():
        setattr(state, k, v)
    return state


class ProberAgent(BaseAgent):
    """Drives one skill probe to a stopping point.

    Constructed fresh per skill probe (mirroring
    `graph.ainvoke(TurnState(...))` in the LangGraph version) -- this is a
    single-shot test driver, not a long-lived conversational agent. Read
    the final transcript back from `.result` after draining the run.
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    skill: dict
    httpx_client: Any  # httpx.AsyncClient
    rpc_url: str = ""
    dialect: str = "0.3"
    extra_headers: dict = {}
    result: dict | None = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = TurnState(
            skill=self.skill,
            httpx_client=self.httpx_client,
            rpc_url=self.rpc_url,
            dialect=self.dialect,
            extra_headers=self.extra_headers,
        )
        state = _apply(state, await send_initial(state))

        while True:
            route = route_after_send(state)
            if route == "poll":
                state = _apply(state, await poll_task(state))
            elif route == "continue":
                state = _apply(state, await send_followup(state))
            else:
                break

        self.result = {k: v for k, v in vars(state).items() if k != "httpx_client"}
        yield Event(author=self.name)
