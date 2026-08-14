"""Offline tests for strands_prober.py -- mocked at the httpx transport
level (same convention as the golden probers', ADK, raw-SDK, and
fasta2a layers' own tests), so these exercise the REAL Strands/a2a-sdk
client code paths against a fake server. No network access or API key
needed.
"""

import asyncio
import json

import httpx
from a2a.client import ClientConfig
from strands.agent.a2a_agent import A2AAgent

from strands_prober import probe_skill_strands

CARD = {
    "name": "Test Agent",
    "description": "test agent",
    "url": "https://x.test/rpc",
    "version": "1.0.0",
    "protocolVersion": "0.3.0",
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
            "kind": "message", "messageId": "m1", "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        }
    return {
        "jsonrpc": "2.0", "id": request_id,
        "result": {"kind": "task", "id": "t1", "contextId": "c1", "status": status},
    }


def _make_agent(handler) -> tuple[A2AAgent, httpx.AsyncClient]:
    hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    agent = A2AAgent(endpoint="https://x.test", client_config=ClientConfig(httpx_client=hc, streaming=False))
    return agent, hc


def test_completed_skill_returns_matching_transcript():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "completed", "done"))

    async def run():
        agent, hc = _make_agent(handler)
        async with hc:
            return await probe_skill_strands(agent, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert result["turn"] == 1


def test_input_required_is_a_legitimate_one_turn_stop_not_a_failed_continuation():
    """Real finding (see module docstring): A2AAgent has no continuation
    mechanism at all, so this prober only ever sends one turn -- an
    input-required response on turn 1 should be a legitimate blocked
    stopping point, not an error."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "input-required", "which color?"))

    async def run():
        agent, hc = _make_agent(handler)
        async with hc:
            return await probe_skill_strands(agent, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["final_state"] == "input-required"
    assert result["turn"] == 1


def test_working_state_reports_no_polling_available():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "working"))

    async def run():
        agent, hc = _make_agent(handler)
        async with hc:
            return await probe_skill_strands(agent, SKILL)

    result = asyncio.run(run())
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "polling" in result["error"]


def test_card_resolution_failure_reported_cleanly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    async def run():
        agent, hc = _make_agent(handler)
        async with hc:
            return await probe_skill_strands(agent, SKILL)

    result = asyncio.run(run())
    assert result["error"] is not None
    assert result["final_state"] is None


def test_stream_async_error_explains_a_dialect_mismatch_error_in_plain_language():
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. A JSON-RPC
    -32601 "method not found" (the same real-world dialect-mismatch shape
    found live against 2s.io) must come back through probe_skill_strands's
    own error field with a plain-language explanation, not just raw text."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body.get("id", "1"),
                "error": {"code": -32601, "message": 'Method not found: "SendMessage".'},
            },
        )

    async def run():
        agent, hc = _make_agent(handler)
        async with hc:
            return await probe_skill_strands(agent, SKILL)

    result = asyncio.run(run())
    assert result["error"] is not None
    assert "doesn't support the current A2A protocol version" in result["error"]


def test_explain_layer_error_covers_the_documented_legacy_well_known_path_finding():
    """Real user-reported confusion: a customer manually checked
    https://p0stman.com/.well-known/agent-card.json in a browser, got a
    404, and was confused why other engines in the same report still
    showed success -- and independently confirmed the same class of
    gap in this layer specifically, quoting Strands' own real raw error
    text verbatim. Must get a plain-language lead-in naming this known
    limitation, not just the raw A2AClientHTTPError text."""
    from strands_prober import _explain_layer_error

    raw = (
        "A2AClientHTTPError: HTTP Error 404: Failed to fetch agent card from "
        "https://p0stman.com/.well-known/agent-card.json: Client error '404 Not Found' for url "
        "'https://p0stman.com/.well-known/agent-card.json' For more information check: "
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404"
    )
    explained = _explain_layer_error(raw)
    assert explained != raw
    assert f"(raw error: {raw})" in explained
    assert "no fallback to the older, legacy path" in explained
    assert "already found this agent's card successfully" in explained


def test_strands_own_agent_result_drops_status_message_text_with_no_artifacts():
    """Real finding, confirmed via this exact mock (see module docstring):
    Strands' own convert_response_to_agent_result() only reads reply text
    from a separate update_event (TaskStatusUpdateEvent/
    TaskArtifactUpdateEvent) -- a plain (Task, None) tuple (how a
    non-streaming message/send response naturally looks) never falls
    back to task.status.message, only task.artifacts. A real agent that
    answers via status.message with no artifacts (this project has found
    several: agent.co-legal.be, MolTrust, InsideOut) would hand a
    Strands caller using the public invoke_async/__call__ API a
    completely empty reply despite answering correctly. This test
    exercises Strands' own public API directly (not strands_prober.py,
    which deliberately bypasses this) to lock in that the gap is real
    and not just a misreading of the source."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "completed", "done"))

    async def run():
        agent, hc = _make_agent(handler)
        async with hc:
            return await agent.invoke_async("hi")

    result = asyncio.run(run())
    assert result.message["content"] == []
    assert result.state == {}
