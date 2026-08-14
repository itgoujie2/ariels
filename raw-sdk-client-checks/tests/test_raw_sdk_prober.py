"""Offline tests for raw_sdk_prober.py -- mocked at the httpx transport
level (same convention as the golden probers' and the ADK layer's own
tests), so these exercise the REAL a2a-sdk client code paths against a
fake server, not a hand-mocked layer. No network access or API key needed
(a real Anthropic key would only be hit for LLM-generated probe/follow-up
text, which none of these fixtures need since every skill has an
"examples" entry).
"""

import asyncio
import json

import httpx
import pytest

from raw_sdk_prober import probe_skill_raw_sdk, resolve_client

CARD = {
    "name": "Test Agent",
    "description": "test agent",
    "url": "https://x.test/rpc",
    "version": "1.0.0",
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


def test_completed_skill_returns_matching_transcript():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "completed", "done"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_raw_sdk(client, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert result["turn"] == 1


def test_input_required_continuation_completes(monkeypatch):
    """Confirmed live (insideout.luthersystems.com): task_id/context_id are
    fields directly on the Message proto, so continuation is just sending
    a new Message with them set -- much simpler than ADK's synthetic
    FunctionCall mechanism. Locks in that raw_sdk_prober.py does this
    correctly and reaches the real terminal state on turn 2."""
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
            return httpx.Response(200, json=_task_response(body.get("id", "1"), "input-required", "which color?"))
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "completed", "done, blue it is"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_raw_sdk(client, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["states"] == ["input-required", "completed"]
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done, blue it is"
    assert result["turn"] == 2


def test_working_state_reports_no_polling_available():
    """The raw client's get_task() polling was never exercised (no
    reachable agent in the current set naturally hits `working`) -- this
    locks in that the gap is reported as a clear, attributed non-error
    rather than silently mis-marked completed."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "working"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_raw_sdk(client, SKILL)

    result = asyncio.run(run())
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "polling" in result["error"] or "get_task" in result["error"]


def test_card_resolution_failure_reported_cleanly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            return await resolve_client("https://x.test", hc)

    client, err = asyncio.run(run())
    assert client is None
    assert err is not None


def test_send_message_error_wraps_with_a_plain_language_explanation():
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. A JSON-RPC
    -32601 "method not found" during send_message() (the same real-world
    dialect-mismatch shape found live against 2s.io) must come back through
    probe_skill_raw_sdk's own error field with a plain-language explanation,
    not just the raw a2a-sdk exception text."""
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
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_raw_sdk(client, SKILL)

    result = asyncio.run(run())
    assert result["error"] is not None
    assert "doesn't support the current A2A protocol version" in result["error"]
    # The real a2a-sdk client raises its own MethodNotFoundError (no numeric
    # JSON-RPC code preserved in the exception text) -- the raw text is
    # still kept, just under a different shape than the golden probers'
    # own hand-rolled RuntimeError.
    assert "MethodNotFoundError" in result["error"]
