"""Regression suite for a2a_wire._normalize() against real-world response
shapes we've actually encountered while probing live agents. Each fixture in
fixtures/*.json is either a real capture (see its "_provenance" field) or an
explicitly-labeled synthetic construction for a spec-legal shape we haven't
observed live yet.

Purpose: catch regressions in our own parsing code without needing network
access to any live agent — the customer's agent under test always runs live
(PLAN.md §5.4), but *our own* normalizer's correctness doesn't require that.
"""

import asyncio

import httpx
import pytest

from app.a2a_wire import UnrecognizedResponseShapeError, _normalize, resolve_target


def _norm(*args, **kwargs):
    """_normalize() is async (it may fall back to an LLM call for genuinely
    unrecognized content shapes) -- none of the fixtures here trigger that
    path, so this just bridges sync test functions to the async call."""
    return asyncio.run(_normalize(*args, **kwargs))


def test_v03_task_completed(load_fixture):
    result = load_fixture("v03_task_completed")
    out = _norm("0.3", result, None, None)
    assert out["state"] == "completed"
    assert out["agent_text"] == "Order 12345 is currently shipped."
    assert out["artifacts"] == ["Order 12345 is currently shipped."]
    assert out["task_id"] == "e9616049-a48a-43c9-8d63-9062eb632c69"


def test_v03_task_input_required(load_fixture):
    result = load_fixture("v03_task_input_required")
    out = _norm("0.3", result, None, None)
    assert out["state"] == "input-required"
    assert "order ID" in out["agent_text"]
    assert out["artifacts"] == []


def test_v03_bare_message(load_fixture):
    result = load_fixture("v03_bare_message")
    out = _norm("0.3", result, fallback_task_id="t1", fallback_context_id="c1")
    assert out["state"] == "completed"
    assert out["agent_text"] == "Quick synchronous reply, no task was ever created."
    assert out["task_id"] == "t1"  # no task ever opened -> falls back to caller's id


def test_v10_task_completed(load_fixture):
    result = load_fixture("v10_task_completed")
    out = _norm("1.0", result, None, None)
    # v1.0 protobuf enum state names get mapped back to v0.3-style casing so
    # the rest of the codebase (assertions.py etc.) never has to care which
    # dialect a given transcript came from.
    assert out["state"] == "completed"
    assert out["agent_text"] == "Request is completed!"
    assert out["artifacts"] == ["Hello, World! I have received your request (hi)"]


def test_v10_bare_message(load_fixture):
    result = load_fixture("v10_bare_message")
    out = _norm("1.0", result, fallback_task_id="t1", fallback_context_id="c1")
    assert out["state"] == "completed"
    assert out["agent_text"] == "Quick synchronous reply, no task was ever created."


def test_v03_artifact_only_falls_back_to_artifact_text(load_fixture):
    """2s.io (real, live, production) leaves status.message empty and puts
    the whole answer in artifacts instead -- both are spec-legal. Without
    the fallback, agent_text would be '' on a fully successful response.
    The artifact has 2 parts: one "text" kind and one "data" kind (the
    structured version of the same result) -- both get extracted."""
    result = load_fixture("v03_artifact_only")
    out = _norm("0.3", result, None, None)
    assert out["state"] == "completed"
    assert out["agent_text"] != ""
    assert "endpoints" in out["agent_text"]
    assert len(out["artifacts"]) == 2


def test_unrecognized_shape_raises_loud(load_fixture):
    """Neither a Task (no 'kind':'task') nor a Message (no 'parts') -- must
    fail loud rather than silently guessing 'bare message' with empty
    content, which would look like a content gap in the target agent
    instead of a coverage gap in our own parser."""
    result = load_fixture("unrecognized_shape")
    with pytest.raises(UnrecognizedResponseShapeError) as exc_info:
        _norm("0.3", result, None, None)
    assert "foo" in str(exc_info.value)  # top-level keys surfaced for debugging

    with pytest.raises(UnrecognizedResponseShapeError):
        _norm("1.0", result, None, None)


def test_resolve_target_flat_url_only_is_v03():
    card = {"url": "https://x.test/a2a", "protocolVersion": "0.3.0"}
    rpc_url, dialect = resolve_target(card, "https://x.test")
    assert rpc_url == "https://x.test/a2a"
    assert dialect == "0.3"


def test_resolve_target_supported_interfaces_only_is_v10():
    card = {"supportedInterfaces": [{"url": "https://x.test/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]}
    rpc_url, dialect = resolve_target(card, "https://x.test")
    assert rpc_url == "https://x.test/a2a"
    assert dialect == "1.0"


def test_resolve_target_prefers_supported_interfaces_over_flat_url():
    """Real bug, real fix: agent.co-legal.be's card has BOTH a flat `url`
    (no top-level protocolVersion, which used to default to '0.3.0') and a
    `supportedInterfaces` entry correctly declaring '1.0'. Picking the flat
    `url` branch first silently misdetected the dialect and broke every
    skill probe (wrong JSON-RPC method name sent, response shape not
    recognized). supportedInterfaces must win when both are present."""
    card = {
        "url": "https://agent.co-legal.be/a2a/jsonrpc",
        "supportedInterfaces": [
            {
                "url": "https://agent.co-legal.be/a2a/jsonrpc",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ],
    }
    rpc_url, dialect = resolve_target(card, "https://agent.co-legal.be")
    assert dialect == "1.0"
    assert rpc_url == "https://agent.co-legal.be/a2a/jsonrpc"


def test_llm_fallback_triggers_when_no_known_field_has_content(monkeypatch):
    """Proves the fallback wiring itself, not any specific real finding:
    deterministic extraction (text/data parts) finds nothing, state is
    'completed' -- must call app.llm.extract_content and use its result.
    Mocked, not a real API call: this test asserts the *mechanism* engages
    correctly, not what any real model would say. A genuinely new field
    shape a live agent invents next (something other than text/data) would
    hit exactly this path in production."""
    from app import llm

    async def fake_extract_content(raw_obj: dict) -> str:
        assert raw_obj["status"]["state"] == "completed"
        return "the real answer, found by the model, not by a hardcoded field path"

    monkeypatch.setattr(llm, "extract_content", fake_extract_content)

    result = {
        "kind": "task",
        "id": "t1",
        "contextId": "c1",
        "status": {
            "state": "completed",
            # Neither "text" nor "data" -- an exotic shape our deterministic
            # extraction doesn't know about, e.g. a future/nonstandard field.
            "message": {"parts": [{"markdown": "**the real answer**"}]},
        },
        "artifacts": [],
    }
    out = _norm("0.3", result, None, None)
    assert out["agent_text"] == "the real answer, found by the model, not by a hardcoded field path"


def test_llm_fallback_not_called_when_deterministic_extraction_succeeds(monkeypatch):
    """The other half of the contract: don't pay for an LLM call when the
    fast path already found real content."""
    from app import llm

    async def fail_if_called(raw_obj: dict) -> str:
        raise AssertionError("LLM fallback should not have been called")

    monkeypatch.setattr(llm, "extract_content", fail_if_called)

    result = {
        "kind": "task",
        "id": "t1",
        "contextId": "c1",
        "status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": "real answer"}]}},
        "artifacts": [],
    }
    out = _norm("0.3", result, None, None)
    assert out["agent_text"] == "real answer"


def test_fetch_card_falls_back_to_legacy_well_known_path():
    """p0stman.com/Zee (real, live production agent) only serves its card at
    the legacy `/.well-known/agent.json` path -- the current
    `/.well-known/agent-card.json` 404s. Same phenomenon as the wire-
    protocol dialect split, one layer earlier: before a single message is
    ever sent, a client hardcoding only the current path can't even find
    the agent. Uses a mocked transport, not a real network call."""
    from app.a2a_wire import fetch_card

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(404)
        if request.url.path == "/.well-known/agent.json":
            return httpx.Response(200, json={"name": "Zee", "url": "https://p0stman.com/api/agent"})
        raise AssertionError(f"unexpected path: {request.url.path}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_card(client, "https://p0stman.com")

    card = asyncio.run(run())
    assert card["name"] == "Zee"


def test_fetch_card_raises_if_no_known_path_works():
    from app.a2a_wire import NoAgentCardFoundError, fetch_card

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_card(client, "https://nonexistent.test")

    with pytest.raises(NoAgentCardFoundError):
        asyncio.run(run())


def test_fetch_card_rejects_jsonrpc_error_envelope_as_not_a_card():
    """insideout.luthersystems.com (real, live) returns HTTP 200 at its
    well-known path, but the body is a JSON-RPC error envelope
    (`{"jsonrpc": "2.0", "error": {...}}`), not a card -- its discovery
    endpoint is misconfigured/not set up. Without this check, fetch_card
    would silently accept that as "the card" and produce a confusing
    downstream failure (empty skills, null name) instead of a clear
    "couldn't discover this agent" error at the point where it's obvious."""
    from app.a2a_wire import NoAgentCardFoundError, fetch_card

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "invalid request"},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_card(client, "https://insideout.test")

    with pytest.raises(NoAgentCardFoundError) as exc_info:
        asyncio.run(run())
    assert "not a valid agent card" in str(exc_info.value)


def test_check_streaming_accepted_true_for_real_sse(load_fixture):
    """Verified live against our own fixture (v0.3, method 'message/stream')
    and agent.co-legal.be (v1.0, 'SendStreamingMessage'): a real streaming
    endpoint responds with 'data: {...}' lines. This is a lightweight
    availability check only -- it does not parse/accumulate the event
    stream, just confirms the endpoint responds to a streaming call at
    all, same diagnostic level as the dialect-acceptance check."""
    from app.a2a_wire import check_streaming_accepted

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"jsonrpc": "2.0", "id": "1", "result": '
            '{"kind": "status-update", "final": false, "status": {"state": "working"}}}\n\n'
        )
        return httpx.Response(200, content=body.encode())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_streaming_accepted(client, "https://x.test", "0.3")

    result = asyncio.run(run())
    assert result["accepted"] is True


def test_check_streaming_accepted_false_for_non_sse_response():
    from app.a2a_wire import check_streaming_accepted

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "error": {"message": "not supported"}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_streaming_accepted(client, "https://x.test", "0.3")

    result = asyncio.run(run())
    assert result["accepted"] is False


def test_send_message_raises_clean_error_when_body_has_neither_result_nor_error():
    """coinrailz.com (real, live): got HTTP 200 with valid JSON that had
    neither "result" nor "error" -- body["result"] crashed with a raw,
    unattributed KeyError: 'result' instead of a clean protocol error.
    Exact live JSON shape wasn't reproducible after the fact (the server's
    response varied by request in a way not pinned down), so this uses a
    constructed example with the same defining property: valid JSON,
    missing both expected top-level keys."""
    from app.a2a_wire import send_message

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await send_message(client, "https://x.test", "0.3", "ping")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run())
    assert "neither 'result' nor 'error'" in str(exc_info.value)


def test_send_message_handles_non_dict_error_shape():
    """Defends the other half of the same assumption: some agents may
    return "error" as a bare string/other shape rather than the standard
    {"code": ..., "message": ...} object -- body['error'].get('code')
    would otherwise crash with 'str' object has no attribute 'get'."""
    from app.a2a_wire import send_message

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "error": "something broke"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await send_message(client, "https://x.test", "0.3", "ping")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run())
    assert "something broke" in str(exc_info.value)


def test_send_message_wraps_non_json_response():
    """theloopbreaker.com (real, live): HTTP 200 with a non-JSON (empty)
    body surfaced as a bare 'Expecting value: line 1 column 1 (char 0)' --
    accurate but gives no hint it's *our* JSON parsing of *their* response
    that failed. Now wrapped with context."""
    from app.a2a_wire import send_message

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await send_message(client, "https://x.test", "0.3", "ping")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run())
    assert "not valid JSON" in str(exc_info.value)


def test_build_get_task_request_v03_uses_flat_id():
    """Per the reference SDK's own v0.3-compat conversions
    (to_compat_get_task_request): TaskQueryParams takes a flat `id` field,
    method `tasks/get`."""
    from app.a2a_wire import _build_get_task_request

    req = _build_get_task_request("0.3", "task-123")
    assert req["method"] == "tasks/get"
    assert req["params"] == {"id": "task-123"}


def test_build_get_task_request_v10_uses_resource_name():
    """Per the reference SDK's own proto (a2a_v0_3.proto message
    GetTaskRequest): v1.0 uses an AIP resource-name style field,
    `name: "tasks/{task_id}"`, method `GetTask` (PascalCase, matching
    SendMessage's naming convention)."""
    from app.a2a_wire import _build_get_task_request

    req = _build_get_task_request("1.0", "task-123")
    assert req["method"] == "GetTask"
    assert req["params"] == {"name": "tasks/task-123"}


def test_get_task_polls_and_normalizes_like_send_message(load_fixture):
    """get_task() should reuse the same shape-driven _normalize() pipeline
    as send_message() -- a tasks/get response is just another Task-shaped
    result, regardless of dialect. Reuses the same real v0.3 completed-task
    fixture as test_v03_task_completed to prove get_task's parsing matches
    send_message's for an identical shape."""
    from app.a2a_wire import get_task

    fixture = load_fixture("v03_task_completed")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": fixture})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await get_task(client, "https://x.test", "0.3", "task-123")

    result = asyncio.run(run())
    assert result["state"] == "completed"


def test_get_task_surfaces_jsonrpc_error_over_generic_http_status():
    """rsperformance.online (real, live): its message/send correctly
    returns `working`, but polling tasks/get for that same task id shortly
    after gets HTTP 404 *with* a well-formed JSON-RPC error body --
    {"error": {"code": -32001, "message": "Task not found or expired; IDs
    are short-lived..."}}. Before reordering _post_rpc to check the body
    for a JSON-RPC error before calling raise_for_status(), this real,
    specific, actionable message was thrown away in favor of a generic,
    uninformative "404 Not Found" from httpx."""
    from app.a2a_wire import get_task

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "jsonrpc": "2.0",
                "id": "2",
                "error": {
                    "code": -32001,
                    "message": "Task not found or expired; IDs are short-lived (see agent card / operator hints for typical retention).",
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await get_task(client, "https://x.test", "1.0", "expired-task")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run())
    assert "-32001" in str(exc_info.value)
    assert "short-lived" in str(exc_info.value)
