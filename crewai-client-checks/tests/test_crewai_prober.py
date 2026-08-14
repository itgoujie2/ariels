"""Offline tests for crewai_prober.py.

Mocked at the `aexecute_a2a_delegation` function level (imported directly
into `crewai_prober`'s module namespace), the same precedent set by the
AG2 layer's tests: `aexecute_a2a_delegation` is a single async function
returning a plain `TaskStateResult` dict, so there's no complex internal
client machinery to fake at a lower level -- what needs regression
coverage is our own transcript-shaping and continuation logic, not
CrewAI's internal HTTP/event handling (already validated live; see
crewai_prober.py's own docstring for the real agents this was confirmed
against).
"""

import asyncio

import crewai_prober
from crewai_prober import MAX_TURNS, probe_skill_crewai

SKILL = {"id": "s", "name": "s", "description": "d", "examples": ["hi"]}


class FakeStatus:
    def __init__(self, value):
        self.value = value


class FakeHistoryItem:
    def __init__(self, context_id=None, task_id=None):
        self.context_id = context_id
        self.task_id = task_id


def test_completed_reads_text_from_result_field(monkeypatch):
    async def fake_call(**kwargs):
        return {"status": FakeStatus("completed"), "result": "the answer", "history": []}

    monkeypatch.setattr(crewai_prober, "aexecute_a2a_delegation", fake_call)

    result = asyncio.run(probe_skill_crewai("https://x.test", SKILL))
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "the answer"


def test_input_required_reads_text_from_error_field_not_result(monkeypatch):
    """Real finding (confirmed live, insideout.luthersystems.com):
    `result` is None while a task is `input_required` -- the actual
    clarifying-question text rides on the `error` field instead. A caller
    that only reads `result` would see empty content."""
    calls = {"n": 0}

    async def fake_call(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "status": FakeStatus("input_required"),
                "result": None,
                "error": "which region do you want this deployed in?",
                "history": [FakeHistoryItem(context_id="ctx1", task_id="task1")],
            }
        return {
            "status": FakeStatus("completed"),
            "result": "done",
            "history": [FakeHistoryItem(context_id="ctx1", task_id="task1")],
        }

    monkeypatch.setattr(crewai_prober, "aexecute_a2a_delegation", fake_call)

    async def fake_followup(*args, **kwargs):
        return "us-east-1"

    monkeypatch.setattr(crewai_prober.llm, "generate_followup", fake_followup)

    result = asyncio.run(probe_skill_crewai("https://x.test", SKILL))
    assert result["error"] is None
    assert result["states"][0] == "input-required"
    assert result["messages"][0] == "which region do you want this deployed in?"
    assert result["final_state"] == "completed"


def test_continuation_passes_back_prior_turns_context_id_and_task_id(monkeypatch):
    """Real finding (confirmed live): continuing requires passing back the
    *prior turn's* task_id (from `history[-1].task_id`), not just
    context_id -- this locks in that we actually extract and thread both
    fields into the next call."""
    seen_kwargs = []

    async def fake_call(**kwargs):
        seen_kwargs.append(kwargs)
        if len(seen_kwargs) == 1:
            return {
                "status": FakeStatus("input_required"),
                "result": None,
                "error": "anything else?",
                "history": [FakeHistoryItem(context_id="ctx-abc", task_id="task-xyz")],
            }
        return {"status": FakeStatus("completed"), "result": "done", "history": []}

    monkeypatch.setattr(crewai_prober, "aexecute_a2a_delegation", fake_call)

    async def fake_followup(*args, **kwargs):
        return "nope, go ahead"

    monkeypatch.setattr(crewai_prober.llm, "generate_followup", fake_followup)

    asyncio.run(probe_skill_crewai("https://x.test", SKILL))
    assert seen_kwargs[0]["context_id"] is None
    assert seen_kwargs[0]["task_id"] is None
    assert seen_kwargs[1]["context_id"] == "ctx-abc"
    assert seen_kwargs[1]["task_id"] == "task-xyz"


def test_failed_state_reads_explanation_from_error_field_not_forced_to_hard_error(monkeypatch):
    """Real finding (confirmed live, insideout.luthersystems.com turn 2):
    a genuine `TaskState.failed` return (not a raised exception) puts its
    explanation in `error` too. This should flow through as a normal
    terminal state with that text as `final_message`, exactly like any
    other layer's agent-declared failure -- not be forced into a
    transcript-level protocol `error`, which would short-circuit
    assertions.py's normal failed-state checks."""
    async def fake_call(**kwargs):
        return {
            "status": FakeStatus("failed"),
            "result": None,
            "error": "internal: a2a task get: secret required",
            "history": [],
        }

    monkeypatch.setattr(crewai_prober, "aexecute_a2a_delegation", fake_call)

    result = asyncio.run(probe_skill_crewai("https://x.test", SKILL))
    assert result["error"] is None
    assert result["final_state"] == "failed"
    assert result["final_message"] == "internal: a2a task get: secret required"


def test_raised_exception_reported_as_clean_attributed_error(monkeypatch):
    async def fake_call(**kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(crewai_prober, "aexecute_a2a_delegation", fake_call)

    result = asyncio.run(probe_skill_crewai("https://x.test", SKILL))
    assert result["error"] == "turn 1: RuntimeError: connection refused"
    assert result["final_state"] is None


def test_raised_exception_explains_the_documented_secret_required_bug_in_plain_language(monkeypatch):
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. This
    layer's own documented finding #3 (insideout.luthersystems.com's
    -32603 "secret required" backend bug on continuation) must come back
    through probe_skill_crewai's own error field with a plain-language
    explanation, not just raw exception text."""
    async def fake_call(**kwargs):
        raise RuntimeError(
            'A2AClientJSONRPCError: -32603 "internal: internal: a2a task get: secret required"'
        )

    monkeypatch.setattr(crewai_prober, "aexecute_a2a_delegation", fake_call)

    result = asyncio.run(probe_skill_crewai("https://x.test", SKILL))
    assert "own backend errored out" in result["error"]
    assert "secret required" in result["error"]


def test_raised_exception_explains_the_documented_legacy_well_known_path_finding_in_plain_language(monkeypatch):
    """Real user-reported confusion: a customer manually checked
    https://p0stman.com/.well-known/agent-card.json in a browser, got a
    404, and was confused why several engines in the same report still
    showed success. Root cause (see module docstring): CrewAI's client
    only tries the current spec's well-known path, no legacy fallback --
    confirmed live, this agent's card is only served at the legacy
    /.well-known/agent.json path. The raw httpx.HTTPStatusError always
    embeds the failing URL, so this must come back through
    probe_skill_crewai's own error field with a plain-language
    explanation naming that specific limitation."""
    async def fake_call(**kwargs):
        # Real shape: httpx.HTTPStatusError's own message always embeds
        # the failing URL -- a plain RuntimeError with the identical
        # text is enough to exercise _explain_layer_error's substring
        # match without needing to construct a real httpx exception
        # (which requires live request/response objects).
        raise RuntimeError(
            "Client error '404 Not Found' for url 'https://p0stman.com/.well-known/agent-card.json'"
        )

    monkeypatch.setattr(crewai_prober, "aexecute_a2a_delegation", fake_call)

    result = asyncio.run(probe_skill_crewai("https://p0stman.com", SKILL))
    assert "no fallback to the older, legacy path" in result["error"]
    assert "already found this agent's card successfully" in result["error"]
    assert "well-known/agent-card.json" in result["error"]


def test_input_required_past_max_turns_reported_as_non_convergence(monkeypatch):
    async def fake_call(**kwargs):
        return {
            "status": FakeStatus("input_required"),
            "result": None,
            "error": "still need more info",
            "history": [FakeHistoryItem(context_id="c", task_id="t")],
        }

    monkeypatch.setattr(crewai_prober, "aexecute_a2a_delegation", fake_call)

    async def fake_followup(*args, **kwargs):
        return "more info"

    monkeypatch.setattr(crewai_prober.llm, "generate_followup", fake_followup)

    result = asyncio.run(probe_skill_crewai("https://x.test", SKILL))
    assert result["turn"] == MAX_TURNS
    assert "did not reach a terminal state" in result["error"]
    assert result["final_state"] == "input-required"
