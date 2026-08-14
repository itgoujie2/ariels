"""Tests for llm.py's defensive handling. No real API calls -- the
Anthropic client is mocked."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import pytest

from app import llm


class _FakeHttpResponse:
    request = None
    headers: dict = {}

    def __init__(self, status_code: int):
        self.status_code = status_code

    def json(self):
        return {"error": {"message": "fake error"}}


def _overloaded_error() -> Exception:
    # Uses whichever retryable error class llm.py itself actually found
    # available in this venv's installed anthropic SDK version (see
    # llm._RETRYABLE_ERRORS's own comment -- OverloadedError doesn't
    # exist in every version this project's 14 engine venvs happen to
    # have installed), so this test exercises a real member of the same
    # tuple _complete() checks against, in whatever environment it runs.
    error_cls = llm._RETRYABLE_ERRORS[0]
    return error_cls(message="overloaded", response=_FakeHttpResponse(529), body=None)


def test_complete_raises_clean_error_on_empty_content(monkeypatch):
    """Found live (rettfrabonden.com, via the content-extraction fallback
    hitting a real, rare edge case): an empty response.content crashed
    resp.content[0] with a raw, unattributed 'list index out of range'
    several layers away from here. Not a retryable condition -- it's a
    shape the model itself returned, not a transient infra failure."""
    fake_response = SimpleNamespace(content=[])
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=fake_response)))
    monkeypatch.setattr(llm, "_get_client", lambda: fake_client)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(llm._complete("prompt"))
    assert "no content blocks" in str(exc_info.value)
    fake_client.messages.create.assert_awaited_once()  # not retried


def test_complete_returns_text_when_content_present(monkeypatch):
    fake_part = SimpleNamespace(text=' "real answer" ')
    fake_response = SimpleNamespace(content=[fake_part])
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=fake_response)))
    monkeypatch.setattr(llm, "_get_client", lambda: fake_client)

    result = asyncio.run(llm._complete("prompt"))
    assert result == "real answer"


def test_complete_retries_transient_error_and_succeeds(monkeypatch):
    """Real finding (product/'s orchestrator, live against
    insideout.luthersystems.com): a transient Anthropic 529 "Overloaded"
    crashed the whole engine subprocess -- more likely than it sounds
    once 13 engines fire concurrent LLM calls at once against the same
    target. _complete() must retry a transient failure and succeed once
    the target recovers, rather than fail on the first error."""
    call_count = {"n": 0}

    async def flaky_create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise _overloaded_error()
        return SimpleNamespace(content=[SimpleNamespace(text="recovered answer")])

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=flaky_create))
    monkeypatch.setattr(llm, "_get_client", lambda: fake_client)
    monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())  # no real delay in tests

    result = asyncio.run(llm._complete("prompt"))
    assert result == "recovered answer"
    assert call_count["n"] == 3


def test_complete_backs_off_exponentially_between_retries(monkeypatch):
    """Locks in the exponential-backoff shape itself, not just that a
    retry eventually happens -- each successive delay should roughly
    double (base * 2**attempt), not be constant or random-only."""
    async def always_fails(*args, **kwargs):
        raise _overloaded_error()

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=always_fails))
    monkeypatch.setattr(llm, "_get_client", lambda: fake_client)

    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0)  # strip jitter for a clean assertion

    with pytest.raises(tuple(llm._RETRYABLE_ERRORS)):
        asyncio.run(llm._complete("prompt"))

    # _MAX_ATTEMPTS attempts -> _MAX_ATTEMPTS - 1 sleeps between them.
    assert len(delays) == llm._MAX_ATTEMPTS - 1
    for i, delay in enumerate(delays):
        assert delay == pytest.approx(llm._BASE_DELAY_S * (2**i))


def test_complete_exhausts_retries_and_raises_last_error(monkeypatch):
    async def always_fails(*args, **kwargs):
        raise _overloaded_error()

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=always_fails))
    monkeypatch.setattr(llm, "_get_client", lambda: fake_client)
    monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())

    with pytest.raises(tuple(llm._RETRYABLE_ERRORS)):
        asyncio.run(llm._complete("prompt"))


def test_complete_does_not_retry_non_retryable_errors(monkeypatch):
    """A bad API key (AuthenticationError) or malformed request
    (BadRequestError) will never succeed no matter how many times it's
    retried -- these must propagate immediately, not burn through the
    whole backoff schedule first."""
    call_count = {"n": 0}

    async def auth_failure(*args, **kwargs):
        call_count["n"] += 1
        raise anthropic.AuthenticationError(message="bad key", response=_FakeHttpResponse(401), body=None)

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=auth_failure))
    monkeypatch.setattr(llm, "_get_client", lambda: fake_client)
    monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())

    with pytest.raises(anthropic.AuthenticationError):
        asyncio.run(llm._complete("prompt"))
    assert call_count["n"] == 1


def test_generate_probe_falls_back_to_generic_message_on_llm_failure(monkeypatch):
    """Only reached once _complete()'s own retries are fully exhausted
    (a sustained/total LLM outage) -- this text is never judged (see
    module docstring), so falling back keeps the actual A2A protocol
    test running instead of crashing the whole probe."""
    monkeypatch.setattr(llm, "_complete", AsyncMock(side_effect=Exception("529 overloaded")))

    result = asyncio.run(llm.generate_probe({"name": "book_flight", "description": "Books a flight"}))
    assert "book_flight" in result
    assert isinstance(result, str) and result


def test_generate_followup_falls_back_to_generic_message_on_llm_failure(monkeypatch):
    monkeypatch.setattr(llm, "_complete", AsyncMock(side_effect=Exception("529 overloaded")))

    result = asyncio.run(llm.generate_followup("what's your name?", {"name": "book_flight", "description": "Books a flight"}))
    assert isinstance(result, str) and result


def test_extract_content_falls_back_to_empty_string_on_llm_failure(monkeypatch):
    """Unlike generate_probe/generate_followup, this is already the
    last-resort content-extraction path -- on failure there's nothing
    left to fall back to but empty content, which callers already
    handle as a legitimate "no content found" outcome, not a crash."""
    monkeypatch.setattr(llm, "_complete", AsyncMock(side_effect=Exception("529 overloaded")))

    result = asyncio.run(llm.extract_content({"some": "object"}))
    assert result == ""


def test_get_client_disables_sdk_internal_retries(monkeypatch):
    """_complete() owns retry/backoff explicitly now -- the SDK's own
    opaque internal retries must be disabled (max_retries=0) so total
    attempt counts and timing aren't split across two independent,
    uncoordinated retry mechanisms."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm._client = None
    client = llm._get_client()
    assert client.max_retries == 0
