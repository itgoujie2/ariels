"""Tests for version_probe. Card inspection needs no network. _classify_error
and probe()'s aggregation are pure/mockable and worth testing directly --
found live (api.sursatech.com) that conflating "401 Unauthorized" with
"doesn't support this dialect" produces a real, misleading report, not just
an imprecise one."""

import asyncio

import httpx

from app.version_probe import _classify_error, card_schema_risks, declared_versions, probe


def test_declared_versions_v03_top_level():
    card = {"protocolVersion": "0.3.0"}
    assert declared_versions(card) == ["0.3.0"]


def test_declared_versions_v10_supported_interfaces():
    card = {"supportedInterfaces": [{"protocolVersion": "1.0", "protocolBinding": "JSONRPC"}]}
    assert declared_versions(card) == ["1.0"]


def test_declared_versions_none_declared():
    assert declared_versions({"name": "no version info"}) == []


def test_card_schema_risk_missing_url_flagged():
    card = {"supportedInterfaces": [{"url": "https://x.test", "protocolBinding": "JSONRPC"}]}
    risks = card_schema_risks(card)
    assert any(r["risk"] == "missing_top_level_url" for r in risks)


def test_card_schema_risk_no_supported_interfaces_flagged():
    card = {"url": "https://x.test"}
    risks = card_schema_risks(card)
    assert any(r["risk"] == "no_supported_interfaces_declared" for r in risks)


def test_card_schema_no_risks_when_both_present():
    card = {
        "url": "https://x.test",
        "supportedInterfaces": [{"url": "https://x.test", "protocolBinding": "JSONRPC"}],
    }
    assert card_schema_risks(card) == []


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://x.test")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_classify_error_401_is_auth_required():
    assert _classify_error(_http_error(401)) == "auth_required"


def test_classify_error_403_is_auth_required():
    assert _classify_error(_http_error(403)) == "auth_required"


def test_classify_error_402_is_payment_required():
    """gpt55.558686.xyz (real, live): gates even a first message/send call
    behind HTTP 402. Same principle as auth_required, different gate --
    unlike ambiguous custom JSON-RPC codes, 402 is a real, standardized
    status worth its own bucket rather than folding into "other"."""
    assert _classify_error(_http_error(402)) == "payment_required"


def test_classify_error_502_is_other_not_auth():
    assert _classify_error(_http_error(502)) == "other"


def test_classify_error_jsonrpc_method_error_is_unsupported():
    assert _classify_error(RuntimeError("A2A error -32601: Method not found")) == "unsupported"


def test_classify_error_jsonrpc_authentication_required_is_auth_required():
    """ai.syln.cn (real, live): error -32001 'Authentication required' --
    unlike HTTP 401/403 there's no standard JSON-RPC code for this, but the
    phrase itself is generic enough to catch without being tailored to one
    agent's specific wording (contrast api.delx.ai's bespoke -32602 with
    custom hint/details fields, deliberately not pattern-matched)."""
    assert (
        _classify_error(RuntimeError("A2A error -32001: Authentication required"))
        == "auth_required"
    )


def test_classify_error_jsonrpc_internal_error_is_internal_error():
    """neva.dt-agent.co.uk (real, live): error -32603 'Internal error'
    wrapping a leaked upstream Anthropic API auth failure ('invalid
    x-api-key') -- their own backend integration is broken, a completely
    different, real finding from "doesn't support this A2A version"."""
    assert (
        _classify_error(
            RuntimeError(
                "A2A error -32603: Error code: 401 - {'type': 'error', 'error': "
                "{'type': 'authentication_error', 'message': 'invalid x-api-key'}}"
            )
        )
        == "internal_error"
    )


def test_classify_error_malformed_response_is_other_not_unsupported():
    """theloopbreaker.com / coinrailz.com (real, live): a non-JSON or
    non-JSON-RPC-shaped response means the endpoint is broken/wrong, not
    "the dialect isn't understood" -- should land in "other", same bucket
    as a 404/405, not be misattributed as a protocol-version gap."""
    assert _classify_error(RuntimeError("Response body was not valid JSON: boom")) == "other"
    assert (
        _classify_error(RuntimeError("JSON-RPC response has neither 'result' nor 'error' (keys: ['x'])"))
        == "other"
    )


def test_classify_error_version_mismatch_code_is_unsupported():
    """api.pictomancer.ai (real, live agent): sending the spec-correct v1.0
    request (SendMessage method + A2A-Version: 1.0 header, confirmed via raw
    curl, not just our own client) gets rejected with error -32009 claiming
    *we* sent v0.3 -- a real bug in their own version-header parsing, not in
    this prober (the identical request without any version pretense at all
    works fine on their v0.3 path). This just locks in that a differently-
    coded JSON-RPC version error (-32009, not the -32601 seen elsewhere)
    still classifies as "unsupported", not "other"."""
    assert (
        _classify_error(
            RuntimeError("A2A error -32009: A2A version '0.3' is not supported by this handler. Expected version '1.0'.")
        )
        == "unsupported"
    )


def test_probe_auth_required_does_not_count_as_no_dialect_accepted():
    """api.sursatech.com (real, live, v1.0-declaring): 0.3 genuinely
    unsupported (method not found), 1.0 blocked by auth. Before this fix,
    `no_dialect_accepted` would have been True (misleadingly implying we
    couldn't talk to it *at all*), when the real story is "one confirmed
    unsupported, one blocked pending credentials." """

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if '"method": "message/send"' in body or "message/send" in body:
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": "1", "error": {"code": -32601, "message": "Method not found"}},
            )
        return httpx.Response(401)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe(client, {"protocolVersion": "1.0"}, "https://x.test")

    result = asyncio.run(run())
    assert result["dialects_tested"]["0.3"]["reason"] == "unsupported"
    assert result["dialects_tested"]["1.0"]["reason"] == "auth_required"
    assert result["auth_required_dialects"] == ["1.0"]
    assert result["accepted_dialects"] == []
    assert result["no_dialect_accepted"] is False  # NOT "no dialect accepted at all" -- it's auth-gated


def test_probe_internal_error_does_not_count_as_no_dialect_accepted():
    """neva.dt-agent.co.uk-style backend outage: a dialect that errors with
    a JSON-RPC -32603 (their own backend broke) is neither confirmed-
    accepted nor confirmed-unsupported -- shouldn't inflate
    no_dialect_accepted into a false "we couldn't talk to it at all"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": "1", "error": {"code": -32603, "message": "Internal error: upstream auth failed"}},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe(client, {"protocolVersion": "1.0"}, "https://x.test")

    result = asyncio.run(run())
    assert result["dialects_tested"]["0.3"]["reason"] == "internal_error"
    assert result["dialects_tested"]["1.0"]["reason"] == "internal_error"
    assert result["internal_error_dialects"] == ["0.3", "1.0"]
    assert result["no_dialect_accepted"] is False


def test_probe_payment_required_does_not_count_as_no_dialect_accepted():
    """gpt55.558686.xyz-style: a dialect gated behind HTTP 402 is neither
    confirmed-accepted nor confirmed-unsupported -- shouldn't inflate
    no_dialect_accepted into a false "we couldn't talk to it at all"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe(client, {"protocolVersion": "1.0"}, "https://x.test")

    result = asyncio.run(run())
    assert result["dialects_tested"]["0.3"]["reason"] == "payment_required"
    assert result["dialects_tested"]["1.0"]["reason"] == "payment_required"
    assert result["payment_required_dialects"] == ["0.3", "1.0"]
    assert result["no_dialect_accepted"] is False


def test_probe_skips_streaming_check_when_card_does_not_claim_it():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "id": "t1", "status": {"state": "completed"}}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe(client, {"protocolVersion": "1.0", "capabilities": {"streaming": False}}, "https://x.test")

    result = asyncio.run(run())
    assert "streaming" not in result


def test_probe_runs_streaming_check_when_card_claims_it():
    """Only bothers when the card actually claims streaming support -- no
    point flagging "streaming didn't work" for an agent that never said
    it would."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if "SendStreamingMessage" in body or "message/stream" in body:
            return httpx.Response(200, content=b'data: {"result": {"kind": "status-update"}}\n\n')
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "id": "t1", "status": {"state": "completed"}}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe(client, {"protocolVersion": "1.0", "capabilities": {"streaming": True}}, "https://x.test")

    result = asyncio.run(run())
    assert result["streaming"]["accepted"] is True


def test_probe_does_not_crash_when_capabilities_is_a_list():
    """metavision.click (real, live): its agent-card.json declares
    "capabilities": ["cve_lookup", "defi_signals", "mcp_tools", "x402_payments"]
    -- a bare list of capability-name strings, not the spec'd object shape
    ({"streaming": bool, ...}). `card.get("capabilities", {}).get("streaming")`
    crashed with "'list' object has no attribute 'get'", an unhandled
    AttributeError propagating all the way out of probe() (run_prober.py's
    main() only wraps the card-fetch step in try/except, not this). Now
    guards with isinstance(capabilities, dict) before treating it as one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "id": "t1", "status": {"state": "completed"}}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe(
                client,
                {"protocolVersion": "1.0", "capabilities": ["cve_lookup", "defi_signals", "mcp_tools", "x402_payments"]},
                "https://x.test",
            )

    result = asyncio.run(run())
    assert "streaming" not in result
