"""Generic A2A prober, built as a LangGraph state machine.

Given ANY target agent's declared skill (name + description + optional
examples — no assumption about domain), this drives one A2A task to a
stopping point: a terminal state (completed/failed/canceled/rejected), a
blocked state (input-required/auth-required) it can't resolve within
max_turns, or an error. It does not know or care what the skill is *about* —
that's the point (see conversation: reference agent must work against
"whatever customer agent", not one hardcoded toy domain).

Talks over `app.a2a_wire` rather than `a2a.client.A2AClient` — a live test
against an unmodified official agent on the current a2a-sdk (1.x) proved the
wire protocol itself has two live dialects (v0.3 and v1.0, see a2a_wire.py's
docstring), and a single pinned SDK client can only speak one of them.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx
from langgraph.graph import StateGraph

from app import llm
from app.a2a_wire import explain_error, get_task, send_message

MAX_TURNS = 4
# Only input-required is resolvable by continuing the chat. auth-required
# means the spec's own out-of-band auth flow is needed — a text reply can't
# satisfy it, so the prober must stop there, not try to talk past it.
CONTINUABLE_STATES = {"input-required"}
# working/submitted mean the task is still processing on the agent's own
# side -- resolvable by polling tasks/get, not by sending it a new chat
# message (that's what CONTINUABLE_STATES is for). Confirmed live
# (rsperformance.online): a real agent replies synchronously with `working`
# and simply never proceeds via message/send alone.
POLLABLE_STATES = {"working", "submitted"}
POLL_DELAY_SECONDS = 1.5


@dataclass
class TurnState:
    skill: dict  # {"id", "name", "description", "examples": [...]}
    httpx_client: httpx.AsyncClient = field(repr=False)
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
    -- which node comes next only depends on the state we just landed in,
    not which node we came from."""
    if state.error:
        return "done"
    if state.final_state in POLLABLE_STATES and state.turn < MAX_TURNS:
        return "poll"
    if state.final_state in CONTINUABLE_STATES and state.turn < MAX_TURNS:
        return "continue"
    return "done"


graph = (
    StateGraph(TurnState)
    .add_node("send_initial", send_initial)
    .add_node("send_followup", send_followup)
    .add_node("poll_task", poll_task)
    .add_edge("__start__", "send_initial")
    .add_conditional_edges(
        "send_initial",
        route_after_send,
        {"continue": "send_followup", "poll": "poll_task", "done": "__end__"},
    )
    .add_conditional_edges(
        "poll_task",
        route_after_send,
        {"continue": "send_followup", "poll": "poll_task", "done": "__end__"},
    )
    .add_conditional_edges(
        "send_followup",
        route_after_send,
        {"continue": "send_followup", "poll": "poll_task", "done": "__end__"},
    )
    .compile()
)
