"""Offline tests for praisonai_prober.py.

Unlike the ADK/raw-SDK/fasta2a/Strands/Agno layers' tests (mocked at the
httpx-transport level), these mock at `A2AProvider._make_request` --
PraisonAI's client uses raw `urllib.request` internally, not `httpx`,
so there's no httpx transport seam to intercept. Patching
`_make_request` still exercises the REAL `invoke()` parsing logic
(result/error extraction) against a controlled response, the same
precedent set by the AG2/CrewAI layers for clients whose internals
don't offer a clean httpx-level mock point.
"""

import asyncio

from praisonai.endpoints.providers.a2a import A2AProvider

from praisonai_prober import probe_skill_praisonai

SKILL = {"id": "s", "name": "s", "description": "d", "examples": ["hi"]}


def _task_response(state: str, text: str | None = None) -> dict:
    status = {"state": state}
    if text is not None:
        status["message"] = {
            "kind": "message", "messageId": "m1", "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        }
    return {"status": 200, "data": {"result": {"kind": "task", "id": "t1", "contextId": "c1", "status": status}}}


def test_completed_skill_returns_matching_transcript(monkeypatch):
    monkeypatch.setattr(A2AProvider, "_make_request", lambda self, method, path, json_data=None: _task_response("completed", "done"))

    provider = A2AProvider(base_url="https://x.test", timeout=30)
    result = asyncio.run(probe_skill_praisonai(provider, SKILL))
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert result["turn"] == 1


def test_input_required_is_a_legitimate_one_turn_stop_not_a_failed_continuation():
    """Real finding (see module docstring): A2AProvider has no
    task_id/context_id field anywhere in its outgoing message, so this
    prober only ever sends one turn -- an input-required response on
    turn 1 should be a legitimate blocked stopping point, not an error."""
    def fake_request(self, method, path, json_data=None):
        return _task_response("input-required", "which color?")

    async def run():
        provider = A2AProvider(base_url="https://x.test", timeout=30)
        provider._make_request = fake_request.__get__(provider)
        return await probe_skill_praisonai(provider, SKILL)

    result = asyncio.run(run())
    assert result["error"] is None
    assert result["final_state"] == "input-required"
    assert result["turn"] == 1


def test_working_state_reports_no_polling_available():
    def fake_request(self, method, path, json_data=None):
        return _task_response("working")

    async def run():
        provider = A2AProvider(base_url="https://x.test", timeout=30)
        provider._make_request = fake_request.__get__(provider)
        return await probe_skill_praisonai(provider, SKILL)

    result = asyncio.run(run())
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "polling" in result["error"]


def test_json_rpc_error_reported_cleanly():
    def fake_request(self, method, path, json_data=None):
        return {"status": 200, "data": {"error": {"code": -32602, "message": "Invalid params"}}}

    async def run():
        provider = A2AProvider(base_url="https://x.test", timeout=30)
        provider._make_request = fake_request.__get__(provider)
        return await probe_skill_praisonai(provider, SKILL)

    result = asyncio.run(run())
    assert result["error"] == "Invalid params"
    assert result["final_state"] is None


def test_invalid_params_error_explains_the_documented_type_vs_kind_finding_in_plain_language():
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. This
    layer's own decisive documented finding (A2AProvider building parts
    with "type" instead of the spec's "kind") surfaces as a real, live
    -32602 "Invalid params" message naming the missing text part
    (confirmed live against 2s.io) -- it must come back through
    probe_skill_praisonai's own error field with a plain-language
    explanation specific to this known limitation, not raw text."""
    def fake_request(self, method, path, json_data=None):
        return {
            "status": 200,
            "data": {
                "error": {
                    "code": -32602,
                    "message": '"message" must contain at least one text part',
                },
            },
        }

    async def run():
        provider = A2AProvider(base_url="https://x.test", timeout=30)
        provider._make_request = fake_request.__get__(provider)
        return await probe_skill_praisonai(provider, SKILL)

    result = asyncio.run(run())
    assert "own outgoing request construction" in result["error"]
    assert "must contain at least one text part" in result["error"]


def test_html_404_body_explains_the_hardcoded_endpoint_finding_in_plain_language():
    """Real user-reported confusion: running against p0stman.com (a
    Next.js site) showed "a bunch of raw code" instead of a clear
    error. Root cause: A2AProvider hardcodes its RPC path to
    "{base_url}/a2a" (this layer's own documented finding #2), which
    doesn't exist on this real target -- the site's own 404 page (full
    HTML) comes back, and PraisonAI's own _make_request() (confirmed by
    reading its source: base.py's HTTPError handler falls back to
    {"message": body} on a JSONDecodeError) puts the *entire raw HTML
    document* into the error message verbatim. Must be detected
    structurally and explained, with the raw HTML truncated rather than
    dumped in full."""
    html_body = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charSet=\"utf-8\"/>"
        + "<script src=\"/_next/static/chunks/webpack.js\"></script>" * 20
        + "<title>Off the path | p0stman</title></head><body>404</body></html>"
    )

    def fake_request(self, method, path, json_data=None):
        return {"status": 404, "error": {"message": html_body}}

    async def run():
        provider = A2AProvider(base_url="https://x.test", timeout=30)
        provider._make_request = fake_request.__get__(provider)
        return await probe_skill_praisonai(provider, SKILL)

    result = asyncio.run(run())
    assert "hardcodes its RPC endpoint" in result["error"]
    assert "non-JSON-body handling" in result["error"]
    # Truncated, not dumped in full -- the report must stay readable.
    assert len(result["error"]) < len(html_body)
    assert "chars total)" in result["error"]


def test_empty_data_from_wrong_hardcoded_endpoint_reported_cleanly_not_crashed():
    """Real finding (see module docstring): when the hardcoded
    {base_url}/a2a path doesn't correspond to a target's real endpoint,
    A2AProvider can come back ok=True with an empty data dict instead of
    a clear error -- confirmed live against insideout.luthersystems.com.
    Locks in that this is surfaced as a clean attributed error rather
    than an unhandled crash."""
    def fake_request(self, method, path, json_data=None):
        return {"status": 200, "data": {}}

    async def run():
        provider = A2AProvider(base_url="https://x.test", timeout=30)
        provider._make_request = fake_request.__get__(provider)
        return await probe_skill_praisonai(provider, SKILL)

    result = asyncio.run(run())
    assert result["error"] is not None
    assert "UnrecognizedResponseShapeError" in result["error"]
    assert result["final_state"] is None


def test_raised_exception_reported_as_clean_attributed_error():
    def fake_request(self, method, path, json_data=None):
        raise RuntimeError("connection refused")

    async def run():
        provider = A2AProvider(base_url="https://x.test", timeout=30)
        provider._make_request = fake_request.__get__(provider)
        return await probe_skill_praisonai(provider, SKILL)

    result = asyncio.run(run())
    assert result["error"] == "turn 1: RuntimeError: connection refused"
    assert result["final_state"] is None
