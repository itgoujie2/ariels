"""Offline tests for fasta2a_prober.py -- mocked at the httpx transport
level (same convention as the golden probers', ADK, and raw-SDK layers'
own tests), so these exercise the REAL fasta2a client code paths
(pydantic validation included) against a fake server. No network access
or API key needed.
"""

import asyncio
import json

import httpx

from fasta2a_prober import _resolve_v03_rpc_url, probe_skill_fasta2a, resolve_client

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


def test_completed_skill_returns_matching_transcript():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "completed", "done"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, _card, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_fasta2a(client, SKILL)

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
            return httpx.Response(200, json=_task_response(body.get("id", "1"), "input-required", "which color?"))
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "completed", "done, blue it is"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, _card, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_fasta2a(client, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["states"] == ["input-required", "completed"]
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done, blue it is"
    assert result["turn"] == 2


def test_working_state_reports_no_polling_available():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "working"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, _card, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_fasta2a(client, SKILL)

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

    client, _card, err = asyncio.run(run())
    assert client is None
    assert err is not None


def test_validation_error_explains_the_documented_strict_schema_finding_in_plain_language():
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. This
    layer's own documented finding #2 (fasta2a's strict pydantic
    validation rejecting real, spec-legal-ish response shapes -- e.g.
    p0stman.com's missing `kind` discriminator) surfaces as a raw
    pydantic ValidationError; it must come back through
    probe_skill_fasta2a's own error field with a plain-language
    explanation specific to this known limitation, not a generic bucket
    or raw text."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        # Missing the required "kind"/"id"/"status" fields entirely --
        # send_message_response_ta's strict TypedDict validation rejects
        # this outright with a pydantic ValidationError, the same shape
        # already confirmed live against p0stman.com/arcasos.com.
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body.get("id", "1"),
            "result": {"nonsense": "not a real task or message"},
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, _card, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_fasta2a(client, SKILL)

    result = asyncio.run(run())
    assert result["error"] is not None
    assert "strict pydantic schema validation" in result["error"]
    assert "ValidationError" in result["error"]


def test_json_rpc_error_reported_cleanly_not_raised():
    """fasta2a's A2AClient only raises on HTTP-status failures -- a
    well-formed JSON-RPC error envelope on a 200 response is returned as a
    normally-parsed dict with an 'error' key instead. Locks in that this
    prober reads that key rather than assuming any 200 response means
    success."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body.get("id", "1"),
            "error": {"code": -32601, "message": "Method not found"},
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, _card, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_fasta2a(client, SKILL)

    result = asyncio.run(run())
    assert "doesn't support the current A2A protocol version" in result["error"]
    assert "A2A error -32601: Method not found" in result["error"]
    assert result["final_state"] is None


def test_resolve_v03_rpc_url_prefers_explicit_v03_jsonrpc_interface():
    """Real finding, confirmed live against insideout.luthersystems.com:
    its card declares both a v1.0 and a v0.3 JSONRPC interface at
    different URLs. fasta2a only ever speaks v0.3 (see module docstring),
    so this must pick the v0.3-declared one specifically, not just the
    first supportedInterfaces match (which app.a2a_wire.resolve_target()
    would pick, since that function prefers the most current dialect --
    the wrong choice for this particular client)."""
    card = {
        "url": "https://x.test/legacy-flat-url/",
        "supportedInterfaces": [
            {"url": "https://x.test/v1/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
            {"url": "https://x.test/v0/", "protocolBinding": "JSONRPC", "protocolVersion": "0.3"},
        ],
    }
    assert _resolve_v03_rpc_url(card, "https://x.test") == "https://x.test/v0/"


def test_resolve_v03_rpc_url_falls_back_to_flat_url_when_no_v03_interface():
    card = {
        "url": "https://x.test/legacy-flat-url/",
        "supportedInterfaces": [
            {"url": "https://x.test/v1/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
        ],
    }
    assert _resolve_v03_rpc_url(card, "https://x.test") == "https://x.test/legacy-flat-url/"
