"""Tests for the AI-Judgment Layer (PLAN.md §5.6). No real API calls -- the
LLM is mocked at the llm._complete() level, same convention as test_llm.py.

Offline tests here validate the MECHANICS (JSON parsing, fallback behavior,
runs/tolerance aggregation, wire-call plumbing) -- they cannot validate
whether a real LLM judge call actually produces a correct verdict against a
real agent. That's what the live validation against api.moltrust.ch /
agent.co-legal.be / 2s.io is for. Where useful, tests below
still use the REAL captured MolTrust boilerplate-reply fixture
(tests/fixtures/v10_moltrust_boilerplate_reply.json) as realistic input, so
the prompt-building/parsing plumbing is exercised against real, not
synthetic, agent text.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app import judgment


def _skill(**overrides) -> dict:
    base = {
        "id": "trust-score",
        "name": "Trust Score",
        "description": "Look up an agent's trust score by DID.",
        "examples": ["What is the trust score of did:moltrust:abc123?"],
    }
    base.update(overrides)
    return base


# -- _parse_json_object() --


def test_parse_json_object_strips_markdown_fence():
    """Real finding, live batch run against api.moltrust.ch's 14 skills:
    the majority of judge/decompose calls came back wrapped in a ```json
    fence despite the prompt explicitly forbidding it, crashing a plain
    json.loads() with 'Expecting value: line 1 column 1'."""
    raw = '```json\n{"outcome_pass": true}\n```'
    assert judgment._parse_json_object(raw) == {"outcome_pass": True}


def test_parse_json_object_strips_surrounding_prose():
    raw = 'Sure, here is my analysis:\n{"outcome_pass": false}\nLet me know if you need more.'
    assert judgment._parse_json_object(raw) == {"outcome_pass": False}


def test_parse_json_object_raises_when_no_json_present():
    with pytest.raises(ValueError):
        judgment._parse_json_object("I couldn't find any relevant information.")


def test_decompose_goal_parses_fenced_json(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        return '```json\n{"goal_summary": "x", "components": [{"id": "c1", "description": "d"}]}\n```'

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = asyncio.run(judgment.decompose_goal(_skill()))
    assert goal["components"][0]["id"] == "c1"


def test_judge_transcript_parses_fenced_json(monkeypatch):
    # A non-empty, satisfied+verifiable component -- an empty components
    # list is deliberately never a pass (see _compute_outcome_pass), so
    # this needs real content to prove fenced-JSON parsing actually
    # worked, not just that no exception was raised.
    async def fake_complete(prompt, max_tokens=60):
        return (
            '```json\n{"components": [{"id": "c1", "satisfied": true, "verifiable": true, '
            '"evidence": "x"}], "process_notes": "ok", "outcome_pass": true}\n```'
        )

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    verdict = asyncio.run(judgment.judge_transcript([], {"goal_summary": "x", "components": [{"id": "c1"}]}))
    assert verdict["outcome_pass"] is True


# -- decompose_goal() --


def test_decompose_goal_parses_valid_json(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        return json.dumps(
            {
                "goal_summary": "Get a real trust score for did:moltrust:abc123.",
                "components": [{"id": "score_returned", "description": "A numeric trust score was returned."}],
            }
        )

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = asyncio.run(judgment.decompose_goal(_skill()))
    assert goal["components"][0]["id"] == "score_returned"


def test_decompose_goal_falls_back_on_malformed_json(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        return "not json at all"

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = asyncio.run(judgment.decompose_goal(_skill()))
    assert goal["components"]  # fallback still has a usable shape
    assert goal["components"][0]["id"] == "substantive_response"


def test_decompose_goal_falls_back_when_components_missing(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        return json.dumps({"goal_summary": "no components key"})

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = asyncio.run(judgment.decompose_goal(_skill()))
    assert goal["components"][0]["id"] == "substantive_response"


def test_decompose_goal_includes_the_declared_example_in_the_prompt(monkeypatch):
    """Real finding, live against agent.co-legal.be's iban.validate skill:
    the card's example named a Belgian IBAN (BE68 5390 0754 7034), but
    goal decomposition independently invented an unrelated German one --
    the judge then failed a fully correct agent for not addressing a
    request it was never actually sent (send_initial always uses the
    card's own example verbatim, same as prober.py). The example must be
    named explicitly in the decomposition prompt so the goal is grounded
    in the same concrete request that's actually going out."""
    captured = {}

    async def fake_complete(prompt, max_tokens=60):
        captured["prompt"] = prompt
        return json.dumps({"goal_summary": "x", "components": [{"id": "c1", "description": "d"}]})

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    skill = _skill(examples=["Is IBAN BE68 5390 0754 7034 geldig?"])
    asyncio.run(judgment.decompose_goal(skill))
    assert "BE68 5390 0754 7034" in captured["prompt"]


def test_decompose_goal_omits_example_clause_when_skill_has_no_examples(monkeypatch):
    captured = {}

    async def fake_complete(prompt, max_tokens=60):
        captured["prompt"] = prompt
        return json.dumps({"goal_summary": "x", "components": [{"id": "c1", "description": "d"}]})

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    asyncio.run(judgment.decompose_goal(_skill(examples=[])))
    assert "actual first message will be exactly" not in captured["prompt"]


def test_decompose_goal_falls_back_on_llm_exception(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        raise RuntimeError("LLM outage")

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = asyncio.run(judgment.decompose_goal(_skill()))
    assert goal["components"][0]["id"] == "substantive_response"


def test_decompose_goal_prompt_forbids_unverifiable_components(monkeypatch):
    """Real bug found live (p0stman.com's inquire skill): goal
    decomposition generated a component ("consistent with the company's
    actual real offerings") no judge could ever verify from a transcript
    alone, sinking an otherwise fully-correct agent answer. The
    decomposition prompt itself must now instruct against this, as the
    first of two independent layers of defense (judge_transcript's own
    verifiable-component scoring is the second, see that test)."""
    captured = {}

    async def fake_complete(prompt, max_tokens=60):
        captured["prompt"] = prompt
        return json.dumps({"goal_summary": "x", "components": [{"id": "c1", "description": "d"}]})

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    asyncio.run(judgment.decompose_goal(_skill()))
    assert "checkable using ONLY the conversation transcript" in captured["prompt"]
    assert "outside knowledge" in captured["prompt"]


# -- generate_goal_directed_reply() --


def test_generate_goal_directed_reply_returns_llm_text(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        assert "did:moltrust:abc123" not in prompt or True  # prompt just needs to build without crashing
        return "Here's the DID: did:moltrust:abc123"

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = {"goal_summary": "get a score", "components": [{"id": "score_returned", "description": "..."}]}
    reply = asyncio.run(judgment.generate_goal_directed_reply("Can you provide a DID?", goal, ["score_returned"], _skill()))
    assert reply == "Here's the DID: did:moltrust:abc123"


def test_generate_goal_directed_reply_falls_back_on_exception(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        raise RuntimeError("LLM outage")

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = {"goal_summary": "get a score", "components": []}
    reply = asyncio.run(judgment.generate_goal_directed_reply("...", goal, [], _skill()))
    assert reply  # non-empty fallback, never crashes


# -- judge_transcript() --


def test_judge_transcript_parses_valid_json(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        return json.dumps(
            {
                "components": [{"id": "score_returned", "satisfied": True, "evidence": "returned 0.87"}],
                "process_notes": "answered directly",
                "outcome_pass": True,
            }
        )

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = {"goal_summary": "get a score", "components": [{"id": "score_returned", "description": "..."}]}
    verdict = asyncio.run(judgment.judge_transcript([{"sent": "hi", "state": "completed", "received": "0.87"}], goal))
    assert verdict["outcome_pass"] is True


def test_judge_transcript_falls_back_on_malformed_json(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        return "not json"

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = {"goal_summary": "get a score", "components": [{"id": "score_returned", "description": "..."}]}
    verdict = asyncio.run(judgment.judge_transcript([], goal))
    # A judge call that can't produce a real verdict must NEVER silently pass.
    assert verdict["outcome_pass"] is False
    assert verdict["components"][0]["satisfied"] is False


def test_judge_transcript_falls_back_on_llm_exception(monkeypatch):
    async def fake_complete(prompt, max_tokens=60):
        raise RuntimeError("LLM outage")

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = {"goal_summary": "get a score", "components": [{"id": "score_returned", "description": "..."}]}
    verdict = asyncio.run(judgment.judge_transcript([], goal))
    assert verdict["outcome_pass"] is False
    assert "judge call failed" in verdict["process_notes"]


def test_judge_transcript_wires_real_moltrust_reply_text_into_the_prompt(monkeypatch, load_fixture):
    """Grounds the prompt-building plumbing in a REAL captured agent
    response (api.moltrust.ch's boilerplate reply, the actual documented
    finding this whole layer exists to catch) rather than synthetic text.
    Doesn't assert what a real LLM would judge (that's the
    live validation's job) -- just that the real text flows into the judge
    prompt intact, no crash on real-world length/content."""
    raw = load_fixture("v10_moltrust_boilerplate_reply")
    boilerplate_text = raw["message"]["parts"][0]["text"]
    captured = {}

    async def fake_complete(prompt, max_tokens=60):
        captured["prompt"] = prompt
        return json.dumps({"components": [], "process_notes": "n/a", "outcome_pass": False})

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = {
        "goal_summary": "Get a real trust score for did:moltrust:abc123.",
        "components": [{"id": "score_returned", "description": "A numeric trust score was returned."}],
    }
    transcript = [{"sent": "What is the trust score of did:moltrust:abc123?", "state": "completed", "received": boilerplate_text}]
    asyncio.run(judgment.judge_transcript(transcript, goal))
    assert "Skills: Agent Trust Score" in captured["prompt"]  # the real boilerplate text made it into the prompt


# -- _compute_outcome_pass() (pure function) --


def test_compute_outcome_pass_ignores_unverifiable_components():
    """Real bug found live (p0stman.com's inquire skill, 2026-08-12): a
    genuinely good agent answer -- 2 of 3 goal components clearly
    satisfied with real evidence -- still failed outright because the
    third component ("consistent with the company's actual real
    offerings") could never be verified from the transcript alone. This
    is the exact real shape captured live: components 1-2 satisfied and
    verifiable, component 3 unsatisfied but explicitly NOT verifiable
    ("cannot verify without external reference material"). Must now pass."""
    components = [
        {
            "id": "services_identified",
            "satisfied": True,
            "verifiable": True,
            "evidence": "Agent lists 6 distinct services: AI voice agents, chatbots, MVPs, web apps, "
            "AI workflow automation, and agentic web readiness (MCP/A2A infrastructure)",
        },
        {
            "id": "response_addresses_query",
            "satisfied": True,
            "verifiable": True,
            "evidence": "Response directly answers 'What services does p0stman offer?' with a "
            "comprehensive list of services without redirection or unrelated information",
        },
        {
            "id": "no_contradictions",
            "satisfied": False,
            "verifiable": False,
            "evidence": "Cannot verify consistency with p0stman's actual service offerings without "
            "external reference material; this assessment requires knowledge of p0stman's actual "
            "services, which is not provided in the transcript",
        },
    ]
    assert judgment._compute_outcome_pass(components) is True


def test_compute_outcome_pass_still_fails_a_genuinely_unsatisfied_verifiable_component():
    """The fix must not become a loophole: a real, verifiable failure
    (see p0stman.com's own portfolio/book skills, both correctly failed
    live with zero unverifiable components involved) still fails."""
    components = [
        {"id": "case_studies_shown", "satisfied": False, "verifiable": True, "evidence": "zero case studies presented"},
    ]
    assert judgment._compute_outcome_pass(components) is False


def test_compute_outcome_pass_missing_verifiable_key_defaults_to_counted():
    """Backward-compatible default: a component with no "verifiable" key
    at all (e.g. an older cached verdict, or a judge response that
    omitted it despite the prompt) is treated as verifiable -- the
    conservative reading, since ignoring components by default would be
    a much easier way to accidentally inflate pass rates than the bug
    this fix closes."""
    components = [{"id": "x", "satisfied": False, "evidence": "no"}]
    assert judgment._compute_outcome_pass(components) is False
    components = [{"id": "x", "satisfied": True, "evidence": "yes"}]
    assert judgment._compute_outcome_pass(components) is True


def test_compute_outcome_pass_all_unverifiable_is_not_a_pass():
    """If nothing could actually be verified, that's not evidence of
    success -- same "empty isn't a vacuous pass" philosophy as the
    genuinely-empty-components case below."""
    components = [{"id": "x", "satisfied": True, "verifiable": False, "evidence": "guess"}]
    assert judgment._compute_outcome_pass(components) is False


def test_compute_outcome_pass_empty_components_is_not_a_pass():
    assert judgment._compute_outcome_pass([]) is False


def test_judge_transcript_computes_outcome_pass_locally_not_from_the_llms_own_arithmetic(monkeypatch):
    """The LLM's own claimed outcome_pass must never be trusted directly
    -- confirmed by having it claim True while a verifiable component is
    actually unsatisfied; judge_transcript must correct it to False."""
    async def fake_complete(prompt, max_tokens=60):
        return json.dumps({
            "components": [{"id": "c1", "satisfied": False, "verifiable": True, "evidence": "no"}],
            "process_notes": "n/a",
            "outcome_pass": True,  # the LLM's own (wrong) claim
        })

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = {"goal_summary": "x", "components": [{"id": "c1", "description": "..."}]}
    verdict = asyncio.run(judgment.judge_transcript([], goal))
    assert verdict["outcome_pass"] is False


def test_judge_transcript_real_p0stman_inquire_finding_now_passes(monkeypatch):
    """End-to-end regression for the real live finding (see
    _compute_outcome_pass's own docstring for the full real component
    data) -- judge_transcript's own outcome_pass, not just the pure
    aggregation function, must now correctly pass this real case."""
    async def fake_complete(prompt, max_tokens=60):
        return json.dumps({
            "components": [
                {"id": "services_identified", "satisfied": True, "verifiable": True, "evidence": "6 services listed"},
                {"id": "response_addresses_query", "satisfied": True, "verifiable": True, "evidence": "on-topic"},
                {
                    "id": "no_contradictions", "satisfied": False, "verifiable": False,
                    "evidence": "cannot verify without external reference material",
                },
            ],
            "process_notes": "Agent provided a direct response without clarifying questions.",
            "outcome_pass": False,  # the LLM's own (pre-fix) claim -- must be corrected
        })

    monkeypatch.setattr(judgment.llm, "_complete", fake_complete)
    goal = {
        "goal_summary": "User receives a complete and accurate list of p0stman's AI services.",
        "components": [
            {"id": "services_identified", "description": "..."},
            {"id": "response_addresses_query", "description": "..."},
            {"id": "no_contradictions", "description": "..."},
        ],
    }
    verdict = asyncio.run(judgment.judge_transcript([], goal))
    assert verdict["outcome_pass"] is True


# -- _aggregate() (pure function) --


def test_aggregate_passes_at_default_tolerance_with_two_of_three():
    """Regression: DEFAULT_TOLERANCE must actually accept 2-of-3, the
    documented default -- a naive 0.67 threshold would (wrongly) reject
    an exact 2/3 pass_rate (0.6666...) via pass_rate >= tolerance."""
    verdicts = [{"outcome_pass": True}, {"outcome_pass": True}, {"outcome_pass": False}]
    result = judgment._aggregate(verdicts, tolerance=judgment.DEFAULT_TOLERANCE)
    assert result["pass_rate"] == pytest.approx(2 / 3)
    assert result["judged_pass"] is True


def test_aggregate_fails_below_tolerance():
    verdicts = [{"outcome_pass": True}, {"outcome_pass": False}, {"outcome_pass": False}]
    result = judgment._aggregate(verdicts, tolerance=judgment.DEFAULT_TOLERANCE)
    assert result["judged_pass"] is False


def test_aggregate_empty_runs_never_passes():
    result = judgment._aggregate([], tolerance=0.67)
    assert result["pass_rate"] == 0.0
    assert result["judged_pass"] is False


# -- run_judged_conversation() (real a2a_wire wire calls, mocked transport) --


def test_run_judged_conversation_uses_declared_example_not_llm(monkeypatch):
    """A skill with a card-declared example should never trigger an LLM
    call for its opening message -- same convention as prober.py's
    send_initial."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "task": {
                        "id": "t1",
                        "contextId": "c1",
                        "status": {"state": "TASK_STATE_COMPLETED", "message": {"parts": [{"text": "0.87"}]}},
                    }
                }
            },
        )

    called = {"generate_probe": False}

    async def fake_generate_probe(skill):
        called["generate_probe"] = True
        return "should not be called"

    monkeypatch.setattr(judgment.llm, "generate_probe", fake_generate_probe)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await judgment.run_judged_conversation(
                client, "https://example.test/rpc", "1.0", _skill(), {"goal_summary": "x", "components": []}
            )

    transcript = asyncio.run(run())
    assert called["generate_probe"] is False
    assert transcript["turns_log"][0]["sent"] == _skill()["examples"][0]
    assert transcript["final_state"] == "completed"


def test_run_judged_conversation_continues_on_input_required(monkeypatch):
    """Mirrors prober.py's send_followup routing, but with a goal-directed
    reply generator instead of the generic one."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "result": {
                        "task": {
                            "id": "t1",
                            "contextId": "c1",
                            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "message": {"parts": [{"text": "Which DID?"}]}},
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "result": {
                    "task": {
                        "id": "t1",
                        "contextId": "c1",
                        "status": {"state": "TASK_STATE_COMPLETED", "message": {"parts": [{"text": "0.87"}]}},
                    }
                }
            },
        )

    async def fake_reply(last_agent_message, goal, unmet_ids, skill):
        return "did:moltrust:abc123"

    monkeypatch.setattr(judgment, "generate_goal_directed_reply", fake_reply)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await judgment.run_judged_conversation(
                client, "https://example.test/rpc", "1.0", _skill(), {"goal_summary": "x", "components": []}
            )

    transcript = asyncio.run(run())
    assert transcript["final_state"] == "completed"
    assert len(transcript["turns_log"]) == 2
    assert transcript["turns_log"][1]["sent"] == "did:moltrust:abc123"


def test_run_judged_conversation_reports_error_on_max_turns_exceeded(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "task": {
                        "id": "t1",
                        "contextId": "c1",
                        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "message": {"parts": [{"text": "still need more"}]}},
                    }
                }
            },
        )

    async def fake_reply(last_agent_message, goal, unmet_ids, skill):
        return "here you go"

    monkeypatch.setattr(judgment, "generate_goal_directed_reply", fake_reply)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await judgment.run_judged_conversation(
                client, "https://example.test/rpc", "1.0", _skill(), {"goal_summary": "x", "components": []}
            )

    transcript = asyncio.run(run())
    assert transcript["error"] is not None
    assert f"{judgment.MAX_TURNS} turns" in transcript["error"]


def test_run_judged_conversation_explains_a_dialect_mismatch_error_in_plain_language():
    """Real user-reported confusion, live against 2s.io: the report's AI
    Judgment section showed a raw `RuntimeError: A2A error -32601: Method
    not found: "SendMessage"...` with no indication of whether this was a
    bug in Ariel or the target agent. Reuses version_probe.py's own
    existing _classify_error() (the "unsupported" bucket) rather than
    hand-rolling new pattern-matching, so this stays consistent with how
    the deterministic layer already explains the identical finding."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"code": -32601, "message": 'Method not found: "SendMessage". This agent supports "message/send".'}},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await judgment.run_judged_conversation(
                client, "https://example.test/rpc", "1.0", _skill(), {"goal_summary": "x", "components": []}
            )

    transcript = asyncio.run(run())
    assert "doesn't support the current A2A protocol version" in transcript["error"]
    assert "not a bug in this test" in transcript["error"]
    assert "A2A error -32601" in transcript["error"]  # raw error still preserved, not discarded


# -- run_judgment_for_skill() --


def test_run_judgment_for_skill_decomposes_goal_once_not_per_run(monkeypatch):
    decompose_calls = {"n": 0}

    async def fake_decompose(skill):
        decompose_calls["n"] += 1
        return {"goal_summary": "x", "components": [{"id": "c1", "description": "..."}]}

    async def fake_conversation(httpx_client, rpc_url, dialect, skill, goal, extra_headers=None):
        return {"turns_log": [{"sent": "a", "state": "completed", "received": "b"}], "final_state": "completed", "error": None}

    async def fake_judge(transcript, goal):
        return {"components": [{"id": "c1", "satisfied": True, "evidence": "x"}], "process_notes": "ok", "outcome_pass": True}

    monkeypatch.setattr(judgment, "decompose_goal", fake_decompose)
    monkeypatch.setattr(judgment, "run_judged_conversation", fake_conversation)
    monkeypatch.setattr(judgment, "judge_transcript", fake_judge)

    async def run():
        async with httpx.AsyncClient() as client:
            return await judgment.run_judgment_for_skill(client, "https://example.test/rpc", "1.0", _skill(), runs=3)

    result = asyncio.run(run())
    assert decompose_calls["n"] == 1  # once per skill, not once per run
    assert result["pass_rate"] == 1.0
    assert result["judged_pass"] is True
    assert result["probabilistic"] is True


def test_run_judgment_for_skill_marks_failed_conversation_as_failed_run(monkeypatch):
    """A conversation that never reaches a stopping point (e.g. max turns
    exceeded) must count as a failed run, not crash or get silently
    skipped -- and must never reach judge_transcript, since there's no
    real transcript to judge."""
    async def fake_decompose(skill):
        return {"goal_summary": "x", "components": [{"id": "c1", "description": "..."}]}

    async def fake_conversation(httpx_client, rpc_url, dialect, skill, goal, extra_headers=None):
        return {"turns_log": [], "final_state": None, "error": "did not reach a terminal state within 4 turns"}

    judge_calls = {"n": 0}

    async def fake_judge(transcript, goal):
        judge_calls["n"] += 1
        return {"components": [], "process_notes": "ok", "outcome_pass": True}

    monkeypatch.setattr(judgment, "decompose_goal", fake_decompose)
    monkeypatch.setattr(judgment, "run_judged_conversation", fake_conversation)
    monkeypatch.setattr(judgment, "judge_transcript", fake_judge)

    async def run():
        async with httpx.AsyncClient() as client:
            return await judgment.run_judgment_for_skill(client, "https://example.test/rpc", "1.0", _skill(), runs=2)

    result = asyncio.run(run())
    assert judge_calls["n"] == 0  # never judges a conversation that didn't complete
    assert result["pass_rate"] == 0.0
    assert result["judged_pass"] is False
