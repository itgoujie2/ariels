"""Offline tests for adkgo_prober.py.

Unlike the Python-only native-client layers' tests (mocked at the
httpx-transport level), these mock at `AdkGoBridge.call()` -- the seam
between our own code and the Go subprocess. A transport-level mock
isn't practical here: the thing under test is a compiled Go binary
speaking newline-JSON over stdio, not an httpx client, so there is no
httpx.MockTransport equivalent to intercept. Mocking at this seam still
exercises the real logic we wrote and are responsible for (the turn
loop, CONTINUABLE_STATES/MAX_TURNS handling, error propagation,
_normalize() integration) -- the real bridge/ADK-Go behavior is already
validated live (see adkgo_prober.py's own docstring), the same
precedent set by the Mastra/AG2/CrewAI/PraisonAI layers for
externally-driven clients without a clean lower-level mock point.
"""

import asyncio

from adkgo_prober import AdkGoBridge, _explain_bridge_error, probe_skill_adkgo

SKILL = {"id": "s", "name": "s", "description": "d", "examples": ["hi"]}


def _task_result(state: str, text: str | None = None) -> dict:
    status = {"state": state}
    if text is not None:
        status["message"] = {
            "kind": "message", "messageId": "m1", "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        }
    return {"ok": True, "response": {"kind": "task", "id": "t1", "contextId": "c1", "status": status}, "taskId": "t1", "contextId": "c1"}


class FakeBridge(AdkGoBridge):
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self._proc = None

    async def call(self, command: dict) -> dict:
        return self._responses.pop(0)

    async def close(self) -> None:
        pass


def test_completed_skill_returns_matching_transcript():
    bridge = FakeBridge([_task_result("completed", "done")])

    result = asyncio.run(probe_skill_adkgo(bridge, SKILL))
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

    result = asyncio.run(probe_skill_adkgo(bridge, SKILL))
    assert result["error"] is None
    assert result["states"] == ["input-required", "completed"]
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done, blue it is"
    assert result["turn"] == 2


def test_working_state_reports_no_polling_available():
    bridge = FakeBridge([_task_result("working")])

    result = asyncio.run(probe_skill_adkgo(bridge, SKILL))
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "polling" in result["error"]


def test_bridge_error_reported_cleanly():
    """Real user-reported confusion, live: the raw Go error text ('agent
    card has no supported interfaces') gave no indication of whether
    this was a bug in Ariel or the target agent. Now wrapped with a
    plain-language explanation (see _explain_bridge_error) -- the raw
    text is preserved afterward, not discarded, for anyone who wants it."""
    bridge = FakeBridge([{"ok": False, "error": "agent card has no supported interfaces"}])

    result = asyncio.run(probe_skill_adkgo(bridge, SKILL))
    assert result["error"].startswith("turn 1: ADK-Go's client requires a card to explicitly declare")
    assert "not an issue with the target agent's card" in result["error"]
    assert "(raw error: agent card has no supported interfaces)" in result["error"]
    assert result["final_state"] is None


def test_raised_exception_reported_cleanly():
    class CrashingBridge(AdkGoBridge):
        def __init__(self):
            self._proc = None

        async def call(self, command: dict) -> dict:
            raise RuntimeError("bridge process crashed")

    result = asyncio.run(probe_skill_adkgo(CrashingBridge(), SKILL))
    assert result["error"] == "turn 1: RuntimeError: bridge process crashed"
    assert result["final_state"] is None


# -- _explain_bridge_error() --


def test_explain_bridge_error_covers_every_documented_finding():
    """All 5 known adkgo_bridge error shapes documented in this module's
    own docstring, confirmed live against real agents -- each should get
    a plain-language lead-in, not just the raw Go error text."""
    cases = {
        "agent card has no supported interfaces": "supportedInterfaces",
        "sender creation failed: no compatible transports found: available transports - [_]": "transport",
        'card parsing failed: unknown security scheme type for bearer: [...]': "security-scheme",
        "unknown event type: [kind taskId contextId status final]": "event parsing",
        "agent card resolution failed: failed to fetch an agent card: card request failed, status: 404 Not Found": "legacy path",
    }
    for raw, expected_topic in cases.items():
        explained = _explain_bridge_error(raw)
        assert explained != raw  # got a real explanation, not just passed through
        assert f"(raw error: {raw})" in explained  # raw text preserved, not discarded
        assert expected_topic in explained


def test_explain_bridge_error_passes_through_unrecognized_errors_unchanged():
    """An error shape neither this layer's own 4 documented bridge
    shapes nor the generic a2a_wire.explain_error() buckets recognize
    must never be dropped or garbled -- falls back to the raw text
    as-is."""
    raw = "some brand-new adkgo_bridge error never documented before"
    assert _explain_bridge_error(raw) == raw


def test_explain_bridge_error_falls_back_to_the_generic_shared_explainer():
    """Part of a cross-cutting fix: an error that isn't one of this
    layer's own 4 documented bridge-specific shapes may still be a
    generic, cross-layer-recognizable one (e.g. a dialect mismatch) --
    this must fall back to a2a_wire.explain_error() instead of showing
    raw text unexplained."""
    raw = 'A2A error -32601: Method not found: "SendMessage".'
    explained = _explain_bridge_error(raw)
    assert "doesn't support the current A2A protocol version" in explained
    assert f"(raw error: {raw})" in explained
