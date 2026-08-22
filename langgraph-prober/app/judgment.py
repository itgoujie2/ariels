"""AI-Judgment Layer (PLAN.md §5.6): Percy-equivalent, explicit opt-in add-on.

Deliberately separate from the Protocol Conformance Layer (assertions.py) and
from prober.py's own multi-turn state machine — this is genuinely probabilistic
LLM judgment of open-ended content correctness, never merged into the
deterministic layer's pass/fail. Build-last per PLAN.md §8: only worth building
once the deterministic prober + native-client layers are solid.

Borrows AWS Strands Evals' `ActorSimulator` pattern: an LLM plays a user with a
*concrete, decomposed* goal (never a vague one — "trust score for DID X
returned" not "help the user check trust"), goal-state tracked per
sub-component, run multiple independent times with a tolerance threshold (same
runs/tolerance philosophy traceix already uses). Closes a real, documented gap:
`content_present_on_completion` (the deterministic check) passes MolTrust's
(api.moltrust.ch) repeated boilerplate reply "coincidentally," since it's
non-empty text — it structurally cannot tell "non-empty" from "actually
answered the question." A goal-directed judgment can.

Framework-agnostic (plain asyncio, no LangGraph dependency) so a future port to
adk-prober is a copy, not a rewrite — same convention as a2a_wire.py/
assertions.py/llm.py. Calls llm._complete() directly (the retry/backoff engine
llm.py's own public functions already wrap) rather than adding new
judgment-specific prompts to llm.py itself, since llm.py is kept byte-identical
across all 14 engine directories and these prompts are specific to this layer.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from app import llm
from app.a2a_wire import explain_error, get_task, send_message

MAX_TURNS = 4
DEFAULT_RUNS = 3
# 2-of-3 at the default run count. Deliberately 0.6, not 0.67: 2/3 as a float
# is 0.6666..., which a naive 0.67 threshold would (wrongly) fail via
# pass_rate >= tolerance -- 0.6 clears exactly 2-of-3 while still failing
# 1-of-3 (0.333) and 0-of-3.
DEFAULT_TOLERANCE = 0.6

# Duplicated from prober.py rather than imported from it -- prober.py pulls in
# LangGraph at module level, and this file's whole point is to stay
# framework-agnostic so it can be copied into adk-prober unchanged.
CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}
POLL_DELAY_SECONDS = 1.5


def _parse_json_object(raw: str) -> dict:
    """Found live (a batch run against api.moltrust.ch's 14 skills): the
    majority of judge_transcript/decompose_goal calls failed with
    `JSONDecodeError: Expecting value: line 1 column 1` despite the prompt
    explicitly saying "ONLY a JSON object, no markdown fences, no
    commentary" -- the model doesn't reliably comply, and llm._complete()'s
    own stripping (whitespace + surrounding quotes only) doesn't touch a
    ```json fence or leading/trailing prose. Same pragmatic fix used
    elsewhere in this codebase for the identical problem: don't fight the
    model's own output shape, extract the JSON object substring (first
    '{' to last '}') regardless of what surrounds it."""
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in LLM output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


async def decompose_goal(skill: dict) -> dict:
    """Turns a skill's own card description/examples into a CONCRETE,
    decomposed goal -- never a vague one. No customer-authored test
    definition exists (PLAN.md §5.3: self-generated from the target's own
    card, same as generate_probe), so this is itself an LLM call. Returns
    {"goal_summary": str, "components": [{"id": str, "description": str}]}.
    Falls back to a single generic component on malformed output -- same
    "never crash the probe over an LLM hiccup" philosophy as
    llm.generate_probe's own fallback, just with a fallback *shape*
    instead of fallback *text*, since callers need the components list to
    exist.

    Real bug found live (agent.co-legal.be's iban.validate skill): when a
    skill declares an example (run_judged_conversation's send_initial uses
    it verbatim, same as prober.py), the goal must be decomposed AROUND
    that same concrete example -- not invented independently. Confirmed
    live before this fix: the card's example named a Belgian IBAN, but the
    independently-invented goal named an unrelated German IBAN, so the
    judge failed a fully correct agent for not addressing a request it was
    never actually sent. The example (when present) is now included in the
    prompt and the goal is required to use its exact concrete details."""
    example = skill["examples"][0] if skill.get("examples") else None
    example_clause = (
        f'The user\'s actual first message will be exactly: "{example}" -- decompose the goal AROUND '
        "this specific request, using the SAME concrete details it names (the same identifier/number/date/"
        "etc.), not different ones you invent.\n\n"
        if example
        else ""
    )
    prompt = (
        "You are designing an evaluation for an AI agent's skill, for interoperability "
        "testing. Given the skill below, define a CONCRETE goal a real user might have -- "
        "not a vague one ('help the user with X'), a specific one with checkable outcomes "
        "('trip booked to Lisbon, dates Mar 3-10, price confirmed').\n\n"
        f"Skill name: {skill['name']}\n"
        f"Skill description: {skill['description']}\n\n"
        f"{example_clause}"
        "CRITICAL: every component must be checkable using ONLY the conversation transcript "
        "itself -- what the agent actually said. Never write a component that requires "
        "outside knowledge a judge wouldn't have from the transcript alone, e.g. whether a "
        "claim is factually accurate about the real world, or consistent with information "
        "about the target that was never shown in the conversation. A judge with no access "
        "to anything except the transcript must be able to mark every component true or "
        "false with confidence.\n\n"
        "Reply with ONLY a JSON object, no markdown fences, no commentary:\n"
        '{"goal_summary": "<one sentence, concrete>", '
        '"components": [{"id": "<short_snake_case_id>", "description": "<what must be true '
        'for this specific part of the goal to count as satisfied, verifiable from the '
        'transcript alone>"}]}\n'
        "Use 2-4 components. Each must be independently checkable from the agent's replies."
    )
    try:
        raw = await llm._complete(prompt, max_tokens=400)
        goal = _parse_json_object(raw)
        if not isinstance(goal, dict) or not goal.get("components"):
            raise ValueError("missing components")
        return goal
    except Exception:
        return {
            "goal_summary": f"Get a real, substantive response to a {skill['name']} request.",
            "components": [
                {
                    "id": "substantive_response",
                    "description": "The agent produced a real, on-topic, non-generic response.",
                }
            ],
        }


async def generate_goal_directed_reply(
    last_agent_message: str, goal: dict, unmet_component_ids: list[str], skill: dict
) -> str:
    """A goal-aware sibling of llm.generate_followup: instead of "keep the
    protocol moving," this is "move toward satisfying the STILL-UNMET parts
    of a concrete goal." Falls back to a generic continuation on LLM
    failure, same rationale as generate_followup's own fallback -- this
    text is simulated-user input, never itself judged."""
    unmet = [c for c in goal.get("components", []) if c["id"] in unmet_component_ids]
    unmet_desc = "; ".join(c["description"] for c in unmet) or "(all components currently believed satisfied)"
    prompt = (
        "You are playing a real user of an AI agent, for interoperability testing. "
        f"Your concrete goal: {goal.get('goal_summary', '')}\n"
        f"Still unmet: {unmet_desc}\n"
        f"The agent's skill is: {skill['name']} — {skill['description']}\n"
        f'The agent just said: "{last_agent_message}"\n\n'
        "Write ONE short, realistic reply that moves toward satisfying the still-unmet "
        "goal components. If it needs an identifier you don't have, invent a plausible-"
        "looking one. Reply with only the message text, no quotes, no explanation."
    )
    try:
        return await llm._complete(prompt)
    except Exception:
        return "Yes, please continue toward that."


async def judge_transcript(transcript: list[dict], goal: dict) -> dict:
    """The actual judgment call: reads the full transcript against the
    decomposed goal, returns per-component satisfaction + evidence, a
    short process-rubric note, and an overall outcome_pass. A failed or
    malformed judge call falls back to outcome_pass=False -- unlike the
    generation-side fallbacks above, a judgment call that can't produce a
    real verdict must never silently count as a pass.

    Real bug found live (p0stman.com's inquire skill): decompose_goal()'s
    own prompt now forbids components requiring outside knowledge (see its
    docstring), but as a second, independent layer of defense -- an LLM
    can still slip one through despite instructions -- each component here
    also carries its own "verifiable" flag: whether THIS judge, given only
    the transcript, could actually confirm or refute it. A genuinely good
    agent answer ("lists 6 real services, directly on-topic") was still
    failing outright because one goal component ("consistent with the
    company's actual real offerings") could never be verified from the
    transcript alone -- the judge itself said so ("cannot verify without
    external reference material") and then still marked it unsatisfied,
    sinking an otherwise-correct outcome. outcome_pass is now computed
    HERE in Python from the verifiable components only (never trusting the
    LLM's own pass/fail arithmetic) -- an unverifiable component can never
    by itself fail an otherwise-fully-satisfied goal. Confirmed live: the
    real fixture below (see the test suite) now passes."""
    components = goal.get("components", [])
    prompt = (
        "You are grading whether an AI agent satisfied a user's concrete goal, based on the "
        "conversation transcript below. Be strict: a vague, generic, or off-topic reply does "
        "NOT satisfy a component just because it's present.\n\n"
        f"Goal: {goal.get('goal_summary', '')}\n"
        f"Components to check:\n{json.dumps(components, indent=2)}\n\n"
        f"Transcript:\n{json.dumps(transcript, indent=2)[:6000]}\n\n"
        "For each component, also decide whether it's VERIFIABLE from this transcript alone -- "
        "false if checking it would require outside knowledge not shown here (e.g. real-world "
        "facts about the target that were never stated in the conversation). An unverifiable "
        "component should still get your best-guess `satisfied` value, but `verifiable: false` "
        "so it isn't held against the agent.\n\n"
        "Reply with ONLY a JSON object, no markdown fences, no commentary:\n"
        '{"components": [{"id": "<component id>", "satisfied": <bool>, "verifiable": <bool>, '
        '"evidence": "<short quote or reason>"}], "process_notes": "<one sentence on HOW the '
        'agent got there, e.g. asked relevant clarifying questions vs. hallucinated>", '
        '"outcome_pass": <bool, true only if every VERIFIABLE component is satisfied>}'
    )
    try:
        raw = await llm._complete(prompt, max_tokens=500)
        verdict = _parse_json_object(raw)
        if not isinstance(verdict, dict) or "outcome_pass" not in verdict:
            raise ValueError("missing outcome_pass")
        verdict["outcome_pass"] = _compute_outcome_pass(verdict.get("components", []))
        return verdict
    except Exception as e:
        return {
            "components": [
                {"id": c["id"], "satisfied": False, "verifiable": True, "evidence": "judge call failed"}
                for c in components
            ],
            "process_notes": f"judge call failed: {type(e).__name__}: {e}",
            "outcome_pass": False,
        }


def _compute_outcome_pass(components: list[dict]) -> bool:
    """Pure, deterministic -- never trust the LLM's own pass/fail
    arithmetic in the same JSON blob it just generated the components in.
    A component missing its own "verifiable" key defaults to True (the
    conservative, backward-compatible reading: assume it counts unless the
    judge explicitly said it can't be checked). No components at all is
    NOT a pass -- an empty list means something upstream already failed
    (see decompose_goal's/judge_transcript's own fallback shapes), never a
    vacuous true."""
    if not components:
        return False
    verifiable = [c for c in components if c.get("verifiable", True)]
    if not verifiable:
        return False
    return all(c.get("satisfied") for c in verifiable)


async def run_judged_conversation(
    httpx_client: httpx.AsyncClient,
    rpc_url: str,
    dialect: str,
    skill: dict,
    goal: dict,
    extra_headers: dict | None = None,
) -> dict:
    """Drives ONE full goal-directed conversation. Reuses the exact same
    wire calls (a2a_wire.send_message/get_task) and the exact same
    poll-vs-continue state routing as prober.py -- only the TEXT of each
    user turn differs (goal-directed here, generic-followup there), so a
    judged conversation exercises the same real wire mechanics the
    deterministic layer already validated, not a parallel reimplementation
    of it."""
    extra_headers = extra_headers or {}
    turns_log: list[dict] = []
    task_id = None
    context_id = None
    final_state = None
    final_message = None
    artifacts: list[str] = []
    error = None
    turn = 0

    try:
        probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)
        while turn < MAX_TURNS:
            if turn == 0:
                sent_text = probe_text
                result = await send_message(
                    httpx_client, rpc_url, dialect, sent_text, extra_headers=extra_headers
                )
            elif final_state in POLLABLE_STATES:
                await asyncio.sleep(POLL_DELAY_SECONDS)
                sent_text = "(polled tasks/get)"
                result = await get_task(
                    httpx_client, rpc_url, dialect, task_id, context_id=context_id, extra_headers=extra_headers
                )
            elif final_state in CONTINUABLE_STATES:
                # Per-component satisfaction is judged once, after the
                # conversation ends (judge_transcript) -- checking it
                # mid-conversation would need its own LLM call per turn
                # for a signal the reply generator doesn't actually need:
                # all components, satisfied or not, are always relevant
                # context for "what should the simulated user say next."
                all_ids = [c["id"] for c in goal.get("components", [])]
                sent_text = await generate_goal_directed_reply(final_message, goal, all_ids, skill)
                result = await send_message(
                    httpx_client,
                    rpc_url,
                    dialect,
                    sent_text,
                    task_id=task_id,
                    context_id=context_id,
                    extra_headers=extra_headers,
                )
            else:
                break

            turn += 1
            task_id = result.get("task_id", task_id)
            context_id = result.get("context_id", context_id)
            final_state = result["state"]
            final_message = result["agent_text"]
            artifacts = artifacts + result["artifacts"]
            turns_log.append({"sent": sent_text, "state": final_state, "received": final_message})
        else:
            error = f"did not reach a terminal state within {MAX_TURNS} turns"
    except Exception as e:
        error = explain_error(f"{type(e).__name__}: {e}")

    return {
        "turns_log": turns_log,
        "final_state": final_state,
        "final_message": final_message,
        "artifacts": artifacts,
        "error": error,
    }


def _aggregate(run_verdicts: list[dict], tolerance: float) -> dict:
    """Pure aggregation, no I/O -- kept separate so the runs/tolerance math
    is directly unit-testable without mocking any LLM/HTTP calls."""
    n = len(run_verdicts)
    passes = sum(1 for v in run_verdicts if v.get("outcome_pass"))
    pass_rate = passes / n if n else 0.0
    return {
        "runs": run_verdicts,
        "pass_rate": pass_rate,
        "judged_pass": pass_rate >= tolerance,
    }


async def run_judgment_for_skill(
    httpx_client: httpx.AsyncClient,
    rpc_url: str,
    dialect: str,
    skill: dict,
    extra_headers: dict | None = None,
    runs: int = DEFAULT_RUNS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Runs the full decompose→converse→judge cycle `runs` independent
    times (an LLM-driven conversation isn't deterministic run-to-run, the
    same reasoning traceix's own runs/tolerance philosophy is built on),
    aggregates a pass_rate and a judged_pass bool at `tolerance`. The goal
    itself is decomposed once per skill, not once per run -- the goal is
    what's being tested against, it shouldn't drift between runs of the
    same skill."""
    goal = await decompose_goal(skill)
    run_verdicts = []
    for _ in range(runs):
        transcript = await run_judged_conversation(httpx_client, rpc_url, dialect, skill, goal, extra_headers)
        if transcript.get("error"):
            run_verdicts.append(
                {
                    "components": [],
                    "process_notes": f"conversation did not complete: {transcript['error']}",
                    "outcome_pass": False,
                }
            )
            continue
        verdict = await judge_transcript(transcript["turns_log"], goal)
        run_verdicts.append(verdict)

    result = _aggregate(run_verdicts, tolerance)
    result["skill_id"] = skill["id"]
    result["goal"] = goal
    result["probabilistic"] = True
    return result
