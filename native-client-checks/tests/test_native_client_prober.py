"""Offline tests for native_client_prober.py -- mocked at the httpx
transport level (same convention as the golden probers' own tests), so
these exercise the REAL RemoteA2aAgent/a2a-sdk code paths against a fake
server, not a hand-mocked ADK layer. No network access or API key needed
(a real Anthropic key would only be hit for LLM-generated probe/follow-up
text, which none of these fixtures need since every skill has an
"examples" entry).
"""

import asyncio

import httpx
import pytest
from google.genai import types as genai_types

from native_client_prober import probe_skill_native

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
        import json
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "completed", "done"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_skill_native("https://x.test/.well-known/agent-card.json", SKILL, client)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert result["turn"] == 1


def test_input_required_continuation_completes(monkeypatch):
    """Confirmed live (insideout.luthersystems.com): input-required is
    surfaced as a synthetic FunctionCall (mock_function_call_for_required_
    user_input); continuing means sending a matching FunctionResponse on
    the next turn. This locks in that native_client_prober.py does that
    correctly and reaches the real terminal state on turn 2."""
    import app.llm as llm_module

    async def fake_followup(*args, **kwargs):
        return "yes, proceed"

    monkeypatch.setattr(llm_module, "generate_followup", fake_followup)

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        import json
        body = json.loads(request.content)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, json=_task_response(body.get("id", "1"), "input-required", "which color?"))
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "completed", "done, blue it is"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_skill_native("https://x.test/.well-known/agent-card.json", SKILL, client)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["states"] == ["input-required", "completed"]
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done, blue it is"
    assert result["turn"] == 2


def test_working_state_reports_no_continuation_available():
    """Confirmed by reading RemoteA2aAgent's source: with its default
    client config (polling=False, streaming=False), a working/submitted
    state gets no synthetic FunctionCall at all -- there is no handle to
    continue it. A real caller using the default, out-of-the-box
    configuration would see this agent as permanently unresponsive; this
    test locks in that the gap is reported as a clear, attributed
    non-error rather than silently mis-marked completed."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        import json
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "working"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_skill_native("https://x.test/.well-known/agent-card.json", SKILL, client)

    result = asyncio.run(run())
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "polling" in result["error"] or "unresponsive" in result["error"]


def test_card_resolution_failure_reported_cleanly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_skill_native("https://x.test/.well-known/agent-card.json", SKILL, client)

    result = asyncio.run(run())
    assert result["error"] is not None
    assert result["final_state"] is None
    # The failure surfaces as an event with error_message set (RemoteA2aAgent's
    # construction is lazy -- it never raises), so it counts as one attempted
    # turn, not zero.
    assert result["turn"] == 1


def test_rpc_error_explains_a_dialect_mismatch_error_in_plain_language():
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. A JSON-RPC
    -32601 "method not found" (the same real-world dialect-mismatch shape
    found live against 2s.io) must come back through probe_skill_native's
    own error field with a plain-language explanation, whether it surfaces
    via a raised exception or via an event's own error_message."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        import json
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
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_skill_native("https://x.test/.well-known/agent-card.json", SKILL, client)

    result = asyncio.run(run())
    assert result["error"] is not None
    assert "doesn't support the current A2A protocol version" in result["error"]
