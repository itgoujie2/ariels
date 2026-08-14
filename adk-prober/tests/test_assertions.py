"""Pure function tests for the deterministic Protocol Conformance Layer.
No network, no LLM, no fixtures -- transcripts are just dicts, exactly the
shape prober.py produces."""

from app.assertions import check_transcript, summarize


def _passed(checks: list[dict], name: str) -> bool:
    return next(c["passed"] for c in checks if c["check"] == name)


def test_completed_with_artifact_passes():
    transcript = {
        "final_state": "completed",
        "turn": 1,
        "artifacts": ["some real content"],
        "final_message": "done",
    }
    checks = check_transcript(transcript)
    assert all(c["passed"] for c in checks)


def test_completed_with_no_artifact_and_no_message_fails_that_check_only():
    transcript = {"final_state": "completed", "turn": 1, "artifacts": [], "final_message": ""}
    checks = check_transcript(transcript)
    assert _passed(checks, "known_task_state")
    assert _passed(checks, "reached_terminal_or_explainable_block")
    assert not _passed(checks, "content_present_on_completion")


def test_completed_with_whitespace_only_content_fails():
    transcript = {"final_state": "completed", "turn": 1, "artifacts": ["   "], "final_message": "   "}
    assert not _passed(check_transcript(transcript), "content_present_on_completion")


def test_completed_with_message_only_no_artifacts_passes():
    """Some real agents (verified live: agent.co-legal.be) put the entire
    answer in status.message and never populate artifacts at all -- that's
    legitimate, not a failure."""
    transcript = {"final_state": "completed", "turn": 1, "artifacts": [], "final_message": "real answer here"}
    assert _passed(check_transcript(transcript), "content_present_on_completion")


def test_failed_with_explanation_passes():
    transcript = {"final_state": "failed", "turn": 2, "artifacts": [], "final_message": "order not found"}
    checks = check_transcript(transcript)
    assert all(c["passed"] for c in checks)


def test_failed_with_empty_explanation_fails_that_check_only():
    transcript = {"final_state": "failed", "turn": 2, "artifacts": [], "final_message": ""}
    checks = check_transcript(transcript)
    assert not _passed(checks, "explanation_present_on_failure")


def test_auth_required_is_a_legitimate_blocked_stop_not_a_failure():
    transcript = {"final_state": "auth-required", "turn": 1, "artifacts": [], "final_message": "auth needed"}
    checks = check_transcript(transcript)
    assert all(c["passed"] for c in checks)
    # No artifact/explanation check applies to a blocked-not-terminal state.
    assert {c["check"] for c in checks} == {"reached_stopping_point", "known_task_state", "reached_terminal_or_explainable_block"}


def test_error_short_circuits_to_single_failing_check():
    transcript = {"error": "did not reach a terminal state within 4 turns", "turn": 4}
    checks = check_transcript(transcript)
    assert len(checks) == 1
    assert checks[0]["check"] == "reached_stopping_point"
    assert not checks[0]["passed"]


def test_unrecognized_state_fails_known_task_state_only():
    transcript = {"final_state": "some-future-state", "turn": 1, "artifacts": [], "final_message": ""}
    checks = check_transcript(transcript)
    assert not _passed(checks, "known_task_state")
    assert not _passed(checks, "reached_terminal_or_explainable_block")


def test_summarize_pass_is_and_of_all_checks():
    good = summarize("skill_a", {"final_state": "completed", "turn": 1, "artifacts": ["x"], "final_message": "x"})
    assert good["pass"] is True

    bad = summarize("skill_b", {"final_state": "completed", "turn": 1, "artifacts": [], "final_message": ""})
    assert bad["pass"] is False
    assert bad["skill_id"] == "skill_b"
