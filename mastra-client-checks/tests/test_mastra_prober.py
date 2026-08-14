"""Offline tests for mastra_prober.py.

Unlike the Python-only native-client layers' tests (mocked at the
httpx-transport level), these mock at `MastraBridge.call()` -- the seam
between our own code and the Node subprocess. A transport-level mock
isn't practical here: the thing under test is a persistent Node.js
process speaking newline-JSON over stdio, not an httpx client, so there
is no httpx.MockTransport equivalent to intercept. Mocking at this seam
still exercises the real logic we wrote and are responsible for (the
turn loop, CONTINUABLE_STATES/MAX_TURNS handling, error propagation,
_normalize() integration) -- the real bridge/A2AAgent behavior is
already validated live (see mastra_prober.py's own docstring), the same
precedent set by the AG2/CrewAI/PraisonAI layers for externally-driven
clients without a clean lower-level mock point.
"""

import asyncio

import pytest

from mastra_prober import MastraBridge, probe_skill_mastra

SKILL = {"id": "s", "name": "s", "description": "d", "examples": ["hi"]}


def _task_result(state: str, text: str | None = None) -> dict:
    status = {"state": state}
    if text is not None:
        status["message"] = {
            "kind": "message", "messageId": "m1", "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        }
    return {"ok": True, "task": {"kind": "task", "id": "t1", "contextId": "c1", "status": status}, "message": None, "resumePayload": None}


class FakeBridge(MastraBridge):
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self._proc = None

    async def call(self, command: dict) -> dict:
        return self._responses.pop(0)

    async def close(self) -> None:
        pass


def test_completed_skill_returns_matching_transcript():
    bridge = FakeBridge([_task_result("completed", "done")])

    result = asyncio.run(probe_skill_mastra(bridge, SKILL))
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert result["turn"] == 1


def test_input_required_continuation_completes(monkeypatch):
    import app.llm as llm_module

    async def fake_followup(*args, **kwargs):
        return "yes, proceed"

    monkeypatch.setattr(llm_module, "generate_followup", fake_followup)

    bridge = FakeBridge([
        _task_result("input-required", "which color?"),
        _task_result("completed", "done, blue it is"),
    ])

    result = asyncio.run(probe_skill_mastra(bridge, SKILL))
    assert result["error"] is None
    assert result["states"] == ["input-required", "completed"]
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done, blue it is"
    assert result["turn"] == 2


def test_working_state_reports_no_polling_available():
    bridge = FakeBridge([_task_result("working")])

    result = asyncio.run(probe_skill_mastra(bridge, SKILL))
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "polling" in result["error"]


def test_bridge_error_reported_cleanly():
    bridge = FakeBridge([{"ok": False, "error": "connection refused"}])

    result = asyncio.run(probe_skill_mastra(bridge, SKILL))
    assert result["error"] == "turn 1: connection refused"
    assert result["final_state"] is None


def test_raised_exception_reported_cleanly():
    class CrashingBridge(MastraBridge):
        def __init__(self):
            self._proc = None

        async def call(self, command: dict) -> dict:
            raise RuntimeError("bridge process crashed")

    result = asyncio.run(probe_skill_mastra(CrashingBridge(), SKILL))
    assert result["error"] == "turn 1: RuntimeError: bridge process crashed"
    assert result["final_state"] is None


def test_bridge_js_crash_explains_the_documented_finding_in_plain_language():
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. This
    layer's own documented finding #4 (raw unhandled JS exceptions from
    the mastra_bridge.mjs subprocess, confirmed live against
    agent.co-legal.be/getamber.dev/postalform.com) must come back
    through probe_skill_mastra's own error field with a plain-language
    explanation specific to this known limitation, not raw JS text."""
    bridge = FakeBridge([{
        "ok": False,
        "error": "TypeError: Cannot read properties of undefined (reading 'message')",
    }])

    result = asyncio.run(probe_skill_mastra(bridge, SKILL))
    assert "raw, unattributed JavaScript TypeError" in result["error"]
    assert "Cannot read properties of undefined" in result["error"]


def test_resolve_bridge_init_crash_explains_the_missing_card_url_finding_in_plain_language():
    """This layer's own documented finding #4, the second variant
    (confirmed live against api.moltrust.ch): a card missing its
    top-level `url` crashes Mastra's own URL construction during init,
    not a clean card-parsing error."""
    from mastra_prober import _explain_layer_error

    raw = "TypeError: Failed to parse URL from undefined"
    explained = _explain_layer_error(raw)
    assert explained != raw
    assert f"(raw error: {raw})" in explained
    assert "crashed constructing a request URL" in explained


def test_explain_layer_error_covers_the_documented_legacy_well_known_path_finding():
    """Real user-reported confusion, same finding already confirmed for
    crewai/langchain4j/strands: this project's client card-fetch only
    tries the current A2A spec's well-known path, no legacy fallback --
    confirmed live against p0stman.com, and by reading @mastra/core's
    own compiled source (getAgentCard() only ever constructs the
    current spec's path). mastra_bridge.mjs's own _formatError() now
    appends MastraA2AError's `.data.url` when present, specifically so
    a card-fetch 404 can be told apart from an unrelated RPC-endpoint
    404 (both raise the identical generic "Remote A2A request failed
    with status 404." message otherwise)."""
    from mastra_prober import _explain_layer_error

    raw = (
        "MastraA2AError: Remote A2A request failed with status 404. "
        "(url: https://p0stman.com/.well-known/agent-card.json)"
    )
    explained = _explain_layer_error(raw)
    assert explained != raw
    assert f"(raw error: {raw})" in explained
    assert "no fallback to the older, legacy path" in explained
    assert "already found this agent's card successfully" in explained


def test_continuation_never_reuses_task_or_context_id_is_not_this_layers_concern():
    """Real finding (see module docstring): Mastra's own resumeGenerate()
    silently starts a brand-new task_id/context_id on every "resume"
    call, confirmed live by comparing raw task.id/contextId across
    turns. This prober doesn't and can't detect that from inside a
    single transcript (the task/context ids aren't part of the
    normalized transcript shape at all) -- it's documented as a finding
    about the client's real behavior, not something this offline test
    can regression-check without re-deriving the live-only evidence.
    This test just locks in that a normal two-turn transcript keeps
    working end-to-end regardless."""
    import app.llm as llm_module

    async def fake_followup(*args, **kwargs):
        return "go ahead"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llm_module, "generate_followup", fake_followup)
        bridge = FakeBridge([
            _task_result("input-required", "confirm?"),
            _task_result("completed", "done"),
        ])
        result = asyncio.run(probe_skill_mastra(bridge, SKILL))
    assert result["turn"] == 2
    assert result["final_state"] == "completed"
