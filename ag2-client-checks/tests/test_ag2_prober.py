"""Offline tests for ag2_prober.py.

Unlike the ADK and raw-SDK layers' tests (mocked at the httpx-transport
level, exercising the real client's internal HTTP/event-parsing code
against a fake server), these mock at the `a_generate_remote_reply`
method level instead. Two reasons: (1) AG2's internal event/task-
completion detection (`compat_send_message`/`stream_chunk_to_compat`/
`_is_event_completed`) has enough internal machinery that hand-crafting
a transport-level mock reliably would risk testing our *guess* at its
shape rather than its real behavior; (2) the real behavior is already
validated live in this module's own docstring (p0stman.com end-to-end,
insideout.luthersystems.com's multi-turn loop across real turns). What
actually needs regression coverage is the wrapper logic we wrote --
`_BoundedAutoReplyAgent`'s turn-counting and "exit" escape hatch, and
`probe_skill_ag2`'s transcript shaping -- not AG2's own internals, which
we don't control and aren't the code under test here.
"""

import asyncio

import pytest

from ag2_prober import MAX_TURNS, _BoundedAutoReplyAgent, probe_skill_ag2

SKILL = {"id": "s", "name": "s", "description": "d", "examples": ["hi"]}


def test_get_human_input_returns_exit_after_max_turns(monkeypatch):
    """Confirmed live (insideout.luthersystems.com): AG2's own internal
    while-True loop for input-required has no iteration limit at all --
    the only sanctioned way to break it is answering "exit". This locks
    in that our override enforces MAX_TURNS via that exact mechanism."""
    import app.llm as llm_module

    async def fake_followup(*args, **kwargs):
        return "a followup"

    monkeypatch.setattr(llm_module, "generate_followup", fake_followup)

    agent = _BoundedAutoReplyAgent(url="https://x.test", name="t")
    agent._init_probe_state(SKILL)

    async def run():
        answers = []
        for _ in range(MAX_TURNS + 2):
            answers.append(await agent.a_get_human_input("Input for `t`\nsome question"))
        return answers

    answers = asyncio.run(run())
    assert answers[:MAX_TURNS] == ["a followup"] * MAX_TURNS
    assert answers[MAX_TURNS] == "exit"
    assert answers[MAX_TURNS + 1] == "exit"
    assert agent._probe_exhausted is True


def test_probe_skill_returns_matching_transcript_on_success(monkeypatch):
    async def fake_reply(self, messages=None, **kwargs):
        return True, {"content": "done"}

    monkeypatch.setattr(_BoundedAutoReplyAgent, "a_generate_remote_reply", fake_reply)

    result = asyncio.run(probe_skill_ag2("https://x.test", SKILL))
    assert result["error"] is None
    assert result["final_state"] == "completed"
    assert result["final_message"] == "done"


def test_probe_skill_reports_exhaustion_cleanly(monkeypatch):
    """If a_get_human_input's own turn counter trips _probe_exhausted
    before AG2's loop returns, that should be reported as "did not
    converge" -- the same semantics as the other layers' MAX_TURNS
    handling -- not silently treated as a normal completion."""
    async def fake_reply(self, messages=None, **kwargs):
        # Simulate AG2 calling a_get_human_input past the limit, then
        # honoring "exit" and returning (True, None) -- exactly what
        # AG2's own source does for that branch.
        for _ in range(MAX_TURNS + 1):
            await self.a_get_human_input("Input for `t`\nsome question")
        return True, None

    monkeypatch.setattr(_BoundedAutoReplyAgent, "a_generate_remote_reply", fake_reply)

    import app.llm as llm_module

    async def fake_followup(*args, **kwargs):
        return "a followup"

    monkeypatch.setattr(llm_module, "generate_followup", fake_followup)

    result = asyncio.run(probe_skill_ag2("https://x.test", SKILL))
    assert result["error"] is not None
    assert "did not reach a terminal state" in result["error"]
    assert result["final_state"] == "input-required"


def test_probe_skill_reports_exception_cleanly(monkeypatch):
    async def fake_reply(self, messages=None, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(_BoundedAutoReplyAgent, "a_generate_remote_reply", fake_reply)

    result = asyncio.run(probe_skill_ag2("https://x.test", SKILL))
    assert result["error"] == "RuntimeError: connection refused"
    assert result["final_state"] is None


def test_probe_skill_explains_a_dialect_mismatch_error_in_plain_language(monkeypatch):
    """Part of a cross-cutting fix: every raw error reaching a customer-
    facing report should be explained, not just shown verbatim. A JSON-RPC
    -32601 "method not found" (the same real-world dialect-mismatch shape
    found live against 2s.io) must come back through probe_skill_ag2's own
    error field with a plain-language explanation, not just raw text."""
    async def fake_reply(self, messages=None, **kwargs):
        raise RuntimeError('A2A error -32601: Method not found: "SendMessage".')

    monkeypatch.setattr(_BoundedAutoReplyAgent, "a_generate_remote_reply", fake_reply)

    result = asyncio.run(probe_skill_ag2("https://x.test", SKILL))
    assert "doesn't support the current A2A protocol version" in result["error"]
    assert "-32601" in result["error"]


def test_probe_skill_reports_ambiguous_no_reply_cleanly(monkeypatch):
    """Confirmed live: AG2's a_generate_remote_reply() collapses several
    real outcomes (e.g. a task that completes with no reply content) into
    the same (True, None) signal its own callers can't further
    distinguish -- documented as a real limitation of AG2's own
    abstraction, not something this layer can see past."""
    async def fake_reply(self, messages=None, **kwargs):
        return True, None

    monkeypatch.setattr(_BoundedAutoReplyAgent, "a_generate_remote_reply", fake_reply)

    result = asyncio.run(probe_skill_ag2("https://x.test", SKILL))
    assert result["error"] is not None
    assert "ambiguous" in result["error"]
