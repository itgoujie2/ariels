"""Protocol Conformance Layer (PLAN.md §5.2): deterministic, zero LLM judgment.

These checks only ever look at task-state mechanics — never at whether the
LLM-generated probe/follow-up text or the agent's reply content was any
*good*. Content-quality judgment is explicitly out of scope here; it belongs
to the separate, opt-in AI-judgment layer (§5.6), not this layer.
"""

from __future__ import annotations

TERMINAL_STATES = {"completed", "failed", "canceled", "rejected"}
BLOCKED_STATES = {"input-required", "auth-required"}
KNOWN_STATES = TERMINAL_STATES | BLOCKED_STATES | {"submitted", "working", "unknown"}


def check_transcript(transcript: dict) -> list[dict]:
    checks = []
    final_state = transcript.get("final_state")
    error = transcript.get("error")

    checks.append(
        {
            "check": "reached_stopping_point",
            "passed": error is None,
            "detail": error or f"stopped at '{final_state}' after {transcript.get('turn')} turn(s)",
        }
    )
    if error:
        return checks

    checks.append(
        {
            "check": "known_task_state",
            "passed": final_state in KNOWN_STATES,
            "detail": final_state,
        }
    )

    checks.append(
        {
            "check": "reached_terminal_or_explainable_block",
            "passed": final_state in TERMINAL_STATES or final_state in BLOCKED_STATES,
            "detail": f"'{final_state}' is {'terminal' if final_state in TERMINAL_STATES else 'blocked-pending-info' if final_state in BLOCKED_STATES else 'not a recognized stopping state'}",
        }
    )

    if final_state == "completed":
        # Some real agents (verified live: 2s.io) put the answer only in
        # artifacts and leave status.message empty; others (verified live:
        # agent.co-legal.be) put a complete, substantive answer only in
        # status.message and never populate artifacts at all. Both are
        # legitimate — the check is "was there real content anywhere,"
        # not "were artifacts specifically used."
        artifacts = transcript.get("artifacts") or []
        final_message = (transcript.get("final_message") or "").strip()
        has_artifact_content = any(a.strip() for a in artifacts)
        checks.append(
            {
                "check": "content_present_on_completion",
                "passed": has_artifact_content or bool(final_message),
                "detail": f"{len(artifacts)} artifact part(s), final_message {'present' if final_message else 'empty'}",
            }
        )

    if final_state in ("failed", "rejected"):
        msg = (transcript.get("final_message") or "").strip()
        checks.append(
            {
                "check": "explanation_present_on_failure",
                "passed": bool(msg),
                "detail": msg or "(empty)",
            }
        )

    return checks


def summarize(skill_id: str, transcript: dict) -> dict:
    checks = check_transcript(transcript)
    return {
        "skill_id": skill_id,
        "turns": transcript.get("turn"),
        "state_sequence": transcript.get("states"),
        "final_state": transcript.get("final_state"),
        "elapsed_ms": round(transcript.get("elapsed_ms", 0), 1),
        "transcript": transcript.get("turns_log"),
        "checks": checks,
        "pass": all(c["passed"] for c in checks),
    }
