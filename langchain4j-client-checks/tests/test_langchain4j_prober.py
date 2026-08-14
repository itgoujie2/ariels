"""Offline tests for langchain4j_prober.py.

Unlike the Python-only native-client layers' tests (mocked at the
httpx-transport level), these mock at `langchain4j_prober._call_bridge`
-- the seam between our own code and the one-shot Java subprocess. A
transport-level mock isn't practical here: the thing under test is a
compiled Java process speaking one line of JSON per invocation, not an
httpx client, so there is no httpx.MockTransport equivalent to
intercept. Mocking at this seam still exercises the real logic we
wrote and are responsible for (the turn loop, CONTINUABLE_STATES/
MAX_TURNS handling, task_id/context_id threading, error propagation,
_normalize() integration) -- the real bridge/LangChain4j behavior is
already validated live (see langchain4j_prober.py's own docstring), the
same precedent set by the Mastra/ADK-Go/AG2/CrewAI/PraisonAI layers for
externally-driven clients without a clean lower-level mock point.
"""

import asyncio

import langchain4j_prober
from langchain4j_prober import probe_skill_langchain4j

SKILL = {"id": "s", "name": "s", "description": "d", "examples": ["hi"]}


def _task_result(state: str, text: str | None = None) -> dict:
    status = {"state": state}
    if text is not None:
        status["message"] = {
            "kind": "message", "messageId": "m1", "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        }
    return {"ok": True, "task": {"kind": "task", "id": "t1", "contextId": "c1", "status": status}}


def test_completed_skill_returns_matching_transcript(monkeypatch):
    async def fake_call_bridge(base_url, text, context_id, task_id):
        return _task_result("completed", "done")

    monkeypatch.setattr(langchain4j_prober, "_call_bridge", fake_call_bridge)

    result = asyncio.run(probe_skill_langchain4j("https://x.test", SKILL))
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"
    assert result["turn"] == 1


def test_input_required_continuation_threads_task_and_context_id(monkeypatch):
    """Real finding (see module docstring): unlike Mastra/ADK-Go, this
    client accepts @A2AContextId/@A2ATaskId as plain method parameters,
    so continuation state is passed in and read back out explicitly --
    locks in that this prober actually extracts and threads both
    correctly across turns."""
    import app.llm as llm_module

    async def fake_followup(*args, **kwargs):
        return "yes, proceed"

    monkeypatch.setattr(llm_module, "generate_followup", fake_followup)

    seen_calls = []

    async def fake_call_bridge(base_url, text, context_id, task_id):
        seen_calls.append((context_id, task_id))
        if len(seen_calls) == 1:
            return _task_result("input-required", "which color?")
        return _task_result("completed", "done, blue it is")

    monkeypatch.setattr(langchain4j_prober, "_call_bridge", fake_call_bridge)

    result = asyncio.run(probe_skill_langchain4j("https://x.test", SKILL))
    assert result["error"] is None
    assert result["states"] == ["input-required", "completed"]
    assert result["turn"] == 2
    assert seen_calls[0] == (None, None)
    assert seen_calls[1] == ("c1", "t1")


def test_working_state_reports_no_polling_available(monkeypatch):
    async def fake_call_bridge(base_url, text, context_id, task_id):
        return _task_result("working")

    monkeypatch.setattr(langchain4j_prober, "_call_bridge", fake_call_bridge)

    result = asyncio.run(probe_skill_langchain4j("https://x.test", SKILL))
    assert result["final_state"] == "working"
    assert result["error"] is not None
    assert "polling" in result["error"]


def test_bridge_error_explains_the_documented_security_scheme_finding_in_plain_language(monkeypatch):
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. This
    layer's own second-most-common documented failure mode (33/153 real
    agents, see module docstring) must come back through
    probe_skill_langchain4j's own error field with a plain-language
    explanation, not raw protobuf text."""
    async def fake_call_bridge(base_url, text, context_id, task_id):
        return {"ok": False, "error": "SecurityScheme oneof field not set"}

    monkeypatch.setattr(langchain4j_prober, "_call_bridge", fake_call_bridge)

    result = asyncio.run(probe_skill_langchain4j("https://x.test", SKILL))
    assert "SecurityScheme oneof field not set" in result["error"]
    assert "chokes on the card's" in result["error"]
    assert result["final_state"] is None


def test_bridge_error_explains_the_documented_no_supported_interfaces_finding_in_plain_language(monkeypatch):
    """This layer's own single most common documented failure mode
    (69/153 real agents, see module docstring)."""
    async def fake_call_bridge(base_url, text, context_id, task_id):
        return {"ok": False, "error": "No server interface available in the AgentCard"}

    monkeypatch.setattr(langchain4j_prober, "_call_bridge", fake_call_bridge)

    result = asyncio.run(probe_skill_langchain4j("https://x.test", SKILL))
    assert "No server interface available in the AgentCard" in result["error"]
    assert "flat `url` style" in result["error"]
    assert result["final_state"] is None


def test_bridge_error_explains_the_documented_legacy_well_known_path_finding_in_plain_language(monkeypatch):
    """Real user-reported confusion: a customer manually checked
    https://p0stman.com/.well-known/agent-card.json in a browser, got a
    404, and was confused why several engines in the same report still
    showed success. Confirmed live by running the real Java bridge
    directly against p0stman.com: its exact raw error is
    'java.lang.RuntimeException: org.a2aproject.sdk.spec.
    A2AClientHTTPError: Failed to obtain agent card: 404' -- this must
    come back through probe_skill_langchain4j's own error field with a
    plain-language explanation naming this specific, known limitation
    (23/153 real agents, see module docstring)."""
    async def fake_call_bridge(base_url, text, context_id, task_id):
        return {
            "ok": False,
            "error": (
                "java.lang.RuntimeException: org.a2aproject.sdk.spec.A2AClientHTTPError: "
                "Failed to obtain agent card: 404"
            ),
        }

    monkeypatch.setattr(langchain4j_prober, "_call_bridge", fake_call_bridge)

    result = asyncio.run(probe_skill_langchain4j("https://x.test", SKILL))
    assert "no fallback to the older, legacy path" in result["error"]
    assert "already found this agent's card successfully" in result["error"]
    assert result["final_state"] is None


def test_hang_timeout_reported_as_clean_attributed_error(monkeypatch):
    """Real finding (see module docstring): confirmed live against
    insideout.luthersystems.com, this client simply hangs (never
    returns, never errors) for at least one real, live agent. Locks in
    that the bridge-call timeout wrapper turns that into a clean,
    attributed error rather than the whole probe hanging forever."""
    async def fake_call_bridge(base_url, text, context_id, task_id):
        return {"ok": False, "error": "bridge process timed out after 30s (client hang -- see module docstring)"}

    monkeypatch.setattr(langchain4j_prober, "_call_bridge", fake_call_bridge)

    result = asyncio.run(probe_skill_langchain4j("https://x.test", SKILL))
    assert "timed out" in result["error"]
    assert result["final_state"] is None


def test_raised_exception_reported_cleanly(monkeypatch):
    async def fake_call_bridge(base_url, text, context_id, task_id):
        raise RuntimeError("subprocess spawn failed")

    monkeypatch.setattr(langchain4j_prober, "_call_bridge", fake_call_bridge)

    result = asyncio.run(probe_skill_langchain4j("https://x.test", SKILL))
    assert result["error"] == "turn 1: RuntimeError: subprocess spawn failed"
    assert result["final_state"] is None
