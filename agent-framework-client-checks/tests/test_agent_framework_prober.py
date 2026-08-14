"""Offline tests for agent_framework_prober.py -- mocked at the httpx
transport level (same convention as the golden probers', ADK, and
raw-SDK layers' own tests), so these exercise the REAL
agent-framework-a2a/a2a-sdk client code paths against a fake server. No
network access or API key needed.
"""

import asyncio
import json

import httpx

from agent_framework_prober import _explain_layer_error, probe_skill_agent_framework, resolve_agent

CARD = {
    "name": "Test Agent",
    "description": "test agent",
    "url": "https://x.test/rpc",
    "version": "1.0.0",
    "protocolVersion": "1.0",
    "supportedInterfaces": [
        {"url": "https://x.test/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
    ],
    "capabilities": {},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [{"id": "s", "name": "s", "description": "d", "examples": ["hi"], "tags": []}],
}

SKILL = {"id": "s", "name": "s", "description": "d", "examples": ["hi"]}


def _task_response(request_id, state: str, text: str | None = None) -> dict:
    status = {"state": state}
    if text is not None:
        status["message"] = {
            "messageId": "m1", "role": "ROLE_AGENT",
            "parts": [{"text": text}],
        }
    return {
        "jsonrpc": "2.0", "id": request_id,
        "result": {"task": {"id": "t1", "contextId": "c1", "status": status}},
    }


def test_completed_skill_returns_matching_transcript():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "TASK_STATE_COMPLETED", "done"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            agent, _card, err = await resolve_agent("https://x.test", hc)
            assert err is None
            return await probe_skill_agent_framework(agent, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert result["turn"] == 1


def test_input_required_continuation_completes(monkeypatch):
    import app.llm as llm_module

    async def fake_followup(*args, **kwargs):
        return "yes, proceed"

    monkeypatch.setattr(llm_module, "generate_followup", fake_followup)

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, json=_task_response(body.get("id", "1"), "TASK_STATE_INPUT_REQUIRED", "which color?"))
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "TASK_STATE_COMPLETED", "done, blue it is"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            agent, _card, err = await resolve_agent("https://x.test", hc)
            assert err is None
            return await probe_skill_agent_framework(agent, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["states"] == ["input-required", "completed"]
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done, blue it is"
    assert result["turn"] == 2


def test_working_state_reports_no_polling_available():
    """Note: this client's `_updates_from_task()` only surfaces an
    in-progress update at all (even with stream=True) when the task
    carries a status message -- a contentless working/submitted state
    with no message produces zero raw_representation events, a further,
    minor wrinkle on top of the module docstring's main finding. Uses a
    working state *with* a status message here, matching how most real
    agents' progress updates actually look."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "TASK_STATE_WORKING", "still working..."))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            agent, _card, err = await resolve_agent("https://x.test", hc)
            assert err is None
            return await probe_skill_agent_framework(agent, SKILL)

    result = asyncio.run(run())
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "background=False" in result["error"] or "polling" in result["error"].lower()


def test_card_resolution_failure_reported_cleanly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            return await resolve_agent("https://x.test", hc)

    agent, _card, err = asyncio.run(run())
    assert agent is None
    assert err is not None


# -- _explain_layer_error() --


def test_explain_layer_error_covers_the_documented_protobuf_parse_error_finding():
    """This module's own documented finding #5/#6 (see docstring):
    strict protobuf schema validation rejects real, spec-legal-ish
    response shapes -- confirmed live against rsperformance.online with
    this exact raw error text. Must get a plain-language lead-in, not
    just the raw protobuf error."""
    raw = 'ParseError: Message type "lf.a2a.v1.SendMessageResponse" has no field named "id"'
    explained = _explain_layer_error(raw)
    assert explained != raw
    assert f"(raw error: {raw})" in explained
    assert "strict protobuf schema validation" in explained


def test_explain_layer_error_passes_through_unrecognized_errors_to_the_generic_explainer():
    """An error shape neither this layer's own dict nor the generic
    a2a_wire.explain_error() buckets recognize must never be dropped or
    garbled -- falls back to the raw text as-is."""
    raw = "some brand-new agent-framework error never documented before"
    assert _explain_layer_error(raw) == raw
