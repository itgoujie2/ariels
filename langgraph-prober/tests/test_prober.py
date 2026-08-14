"""Tests for prober.py's task-polling behavior.

Found via rsperformance.online (real, live): message/send can return a
non-terminal `working` state on the very first response, with no further
progress possible via message/send alone -- the only way to see the task
finish is polling tasks/get by id. Before poll_task existed, `working` fell
through route_after_send's final `return "done"`, so the prober stopped
immediately and reported a false "not a recognized stopping state" failure
on an agent that was actually still working normally.
"""

import asyncio

import httpx
import pytest

from app.prober import CONTINUABLE_STATES, POLLABLE_STATES, TurnState, graph, route_after_send


def test_pollable_and_continuable_states_are_disjoint():
    """working/submitted (poll) and input-required (chat) need genuinely
    different resolutions -- an agent stuck in one should never be routed
    through the other's mechanism."""
    assert POLLABLE_STATES.isdisjoint(CONTINUABLE_STATES)


def test_route_after_send_polls_on_working():
    state = TurnState(skill={"id": "s"}, httpx_client=None, final_state="working", turn=1)
    assert route_after_send(state) == "poll"


def test_route_after_send_continues_on_input_required():
    state = TurnState(skill={"id": "s"}, httpx_client=None, final_state="input-required", turn=1)
    assert route_after_send(state) == "continue"


def test_route_after_send_stops_on_working_past_max_turns():
    state = TurnState(skill={"id": "s"}, httpx_client=None, final_state="working", turn=4)
    assert route_after_send(state) == "done"


def test_graph_polls_working_task_until_completed(monkeypatch):
    """End-to-end: send_initial gets `working`, poll_task polls tasks/get
    (working, then completed), and the graph correctly ends on the real
    terminal state instead of stopping at the first `working` reply."""
    import app.prober as prober_module

    monkeypatch.setattr(prober_module, "POLL_DELAY_SECONDS", 0)

    poll_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if "tasks/get" in body:
            poll_count["n"] += 1
            if poll_count["n"] < 2:
                state = "working"
                message = None
            else:
                state = "completed"
                message = {"role": "agent", "parts": [{"kind": "text", "text": "done"}]}
            status = {"state": state}
            if message:
                status["message"] = message
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "id": "t1", "status": status}}
            )
        # initial message/send
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "id": "t1", "status": {"state": "working"}}},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await graph.ainvoke(
                TurnState(
                    skill={"id": "s", "name": "s", "description": "d", "examples": ["do the thing"]},
                    httpx_client=client,
                    rpc_url="https://x.test",
                    dialect="0.3",
                )
            )

    result = asyncio.run(run())
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert poll_count["n"] == 2
    assert result["states"] == ["working", "working", "completed"]


def test_send_initial_wraps_a_raw_error_with_a_plain_language_explanation():
    """Real user-reported confusion, live against 2s.io: a raw
    `RuntimeError: A2A error -32601: Method not found: "SendMessage"...`
    landed straight in the transcript's own `error` field with no
    indication of whether this was a bug in Ariel or the target agent.
    send_initial/send_followup/poll_task all wrap via a2a_wire.explain_error()
    now -- this covers send_initial, the one every skill probe hits first."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "error": {
                    "code": -32601,
                    "message": 'Method not found: "SendMessage". This agent supports "message/send".',
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await graph.ainvoke(
                TurnState(
                    skill={"id": "s", "name": "s", "description": "d", "examples": ["do the thing"]},
                    httpx_client=client,
                    rpc_url="https://x.test",
                    dialect="1.0",
                )
            )

    result = asyncio.run(run())
    assert "doesn't support the current A2A protocol version" in result["error"]
    assert "-32601" in result["error"]
