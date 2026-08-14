"""Offline tests for agno_prober.py -- mocked at the httpx transport
level (same convention as the golden probers', ADK, raw-SDK, fasta2a,
and Strands layers' own tests), so these exercise the REAL Agno client
code paths against a fake server. No network access or API key needed.
"""

import asyncio
import json

import httpx

from agno_prober import _resolve_v03_rpc_url, probe_skill_agno, resolve_client

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
            return await probe_skill_agno(client, hc, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert result["turn"] == 1


def test_input_required_is_a_legitimate_one_turn_stop_not_a_failed_continuation():
    """Real finding (see module docstring): A2AClient's public
    send_message() has no task_id parameter at all, so this prober only
    ever sends one turn -- an input-required response on turn 1 should
    be a legitimate blocked stopping point, not an error."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content)
        return httpx.Response(200, json=_task_response(body.get("id", "1"), "input-required", "which color?"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, _card, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_agno(client, hc, SKILL)

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
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, _card, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_agno(client, hc, SKILL)

    result = asyncio.run(run())
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "polling" in result["error"]


def test_card_resolution_failure_reported_cleanly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            return await resolve_client("https://x.test", hc)

    client, _card, err = asyncio.run(run())
    assert client is None
    assert err is not None


def test_send_message_404_explains_the_documented_rest_default_quirk_in_plain_language():
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. A 404
    surfacing from probe_skill_agno's own send call (this prober always
    uses json-rpc mode, so a real 404 here means the resolved endpoint
    itself is wrong) should still come back with plain-language context
    about this layer's own documented REST-mode-default quirk (finding
    #1), not just a raw HTTPStatusError."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("agent-card.json"):
            return httpx.Response(200, json=CARD)
        return httpx.Response(404, text="Not Found")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            client, _card, err = await resolve_client("https://x.test", hc)
            assert err is None
            return await probe_skill_agno(client, hc, SKILL)

    result = asyncio.run(run())
    assert result["error"] is not None
    assert "JSON-RPC mode" in result["error"]
    assert "404" in result["error"]


def test_json_rpc_error_reported_cleanly():
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
            return await probe_skill_agno(client, hc, SKILL)

    result = asyncio.run(run())
    assert "doesn't support the current A2A protocol version" in result["error"]
    assert "A2A error -32601: Method not found" in result["error"]
    assert result["final_state"] is None


def test_resolve_v03_rpc_url_prefers_explicit_v03_jsonrpc_interface():
    """Real finding, confirmed live against insideout.luthersystems.com:
    its card declares both a v1.0 and a v0.3 JSONRPC interface at
    different URLs. Agno's client is v0.3-only (see module docstring),
    so this must pick the v0.3-declared one specifically -- the same
    fix already built for the fasta2a layer, needed again here."""
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


def test_agno_client_default_rest_protocol_404s_against_generic_agent(monkeypatch):
    """Real finding (see module docstring): Agno's A2AClient defaults to
    protocol="rest", which posts to {base_url}/v1/message:send -- a path
    specific to Agno's own server, not a generic A2A agent. Confirmed
    live against 2s.io; this locks in the same behavior offline against
    an agent that only implements the plain JSON-RPC root endpoint."""
    import agno.client.a2a.client as agno_client_module
    from agno.client.a2a.client import A2AClient

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://x.test/rpc/v1/message:send":
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, json=_task_response("1", "completed", "done"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc:
            # Agno's client.py does `from agno.utils.http import
            # get_default_async_client`, binding the name directly into
            # its own module namespace -- patching the original
            # agno.utils.http module has no effect on that already-bound
            # reference, so the patch target must be the imported name
            # inside agno.client.a2a.client itself.
            monkeypatch.setattr(agno_client_module, "get_default_async_client", lambda: hc)
            client = A2AClient(base_url="https://x.test/rpc", timeout=30)  # default protocol="rest"
            return await client.send_message("hi")

    try:
        asyncio.run(run())
        assert False, "expected an HTTPStatusError from the REST-style 404"
    except httpx.HTTPStatusError as e:
        assert e.response.status_code == 404
