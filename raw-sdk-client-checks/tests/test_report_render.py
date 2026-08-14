"""Offline tests for report_render.py -- the standalone, single-engine
HTML report used when this open-source prober is run directly, with no
hosted product involved. Same well-formed-HTML verification convention
already established in product/tests/test_render_detail.py.
"""
from __future__ import annotations

from html.parser import HTMLParser

from app.report_render import render

SAMPLE_SKILL_PASS = {
    "skill_id": "ask",
    "turns": 1,
    "state_sequence": ["completed"],
    "final_state": "completed",
    "elapsed_ms": 1234.5,
    "transcript": [
        {"sent": "What can you help with?", "state": "completed", "received": "I can answer questions about your account."}
    ],
    "checks": [
        {"check": "known_task_state", "passed": True, "detail": "completed"},
        {"check": "content_present_on_completion", "passed": True, "detail": "final_message present"},
    ],
    "pass": True,
}

SAMPLE_SKILL_FAIL = {
    "skill_id": "checkout",
    "turns": 2,
    "state_sequence": ["input-required", "failed"],
    "final_state": "failed",
    "elapsed_ms": 5678.0,
    "transcript": [
        {"sent": "I want to buy item 42", "state": "input-required", "received": "What's your shipping address?"},
        {"sent": "123 Main St", "state": "failed", "received": "Internal error"},
    ],
    "checks": [
        {"check": "known_task_state", "passed": True, "detail": "failed"},
        {"check": "explanation_present_on_failure", "passed": False, "detail": "no explanation text"},
    ],
    "pass": False,
}

SAMPLE_RESULT = {
    "target": "https://example.com/agent",
    "agent_name": "Example Agent",
    "declared_skills": ["ask", "checkout"],
    "skills_tested": [SAMPLE_SKILL_PASS, SAMPLE_SKILL_FAIL],
    "overall_pass": False,
}


class _WellFormedChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.errors: list[str] = []

    def error(self, message):  # pragma: no cover - HTMLParser calls this on fatal errors
        self.errors.append(message)


def test_render_produces_well_formed_html_with_no_stray_braces():
    html = render(SAMPLE_RESULT, engine_label="Test Engine")
    assert "{{" not in html
    assert "}}" not in html
    checker = _WellFormedChecker()
    checker.feed(html)
    assert checker.errors == []


def test_render_shows_engine_label_and_target():
    html = render(SAMPLE_RESULT, engine_label="LangGraph golden prober")
    assert "LangGraph golden prober" in html
    assert "example.com/agent" in html
    assert "Example Agent" in html


def test_render_shows_pass_and_fail_badges():
    html = render(SAMPLE_RESULT, engine_label="Test Engine")
    assert "ask" in html
    assert "checkout" in html
    assert html.count('<details class="card"') == 2
    assert '<span class="badge good">pass</span>' in html
    assert '<span class="badge bad">fail</span>' in html


def test_render_shows_stat_row_with_correct_pass_count():
    html = render(SAMPLE_RESULT, engine_label="Test Engine")
    assert '<div class="num">1/2</div>' in html


def test_render_top_level_error_shows_notice_not_a_crash():
    result = {
        "target": "https://example.com/agent",
        "overall_pass": False,
        "error": "NoAgentCardFoundError: no known well-known path returned a valid agent card",
    }
    html = render(result, engine_label="Test Engine")
    assert "Could not complete this check" in html
    assert "NoAgentCardFoundError" in html
    checker = _WellFormedChecker()
    checker.feed(html)
    assert checker.errors == []


def test_render_escapes_html_in_transcript_content():
    skill = {
        **SAMPLE_SKILL_PASS,
        "transcript": [{"sent": "<script>alert(1)</script>", "state": "completed", "received": "ok"}],
    }
    result = {**SAMPLE_RESULT, "skills_tested": [skill]}
    html = render(result, engine_label="Test Engine")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_shows_check_tooltip_explanations():
    html = render(SAMPLE_RESULT, engine_label="Test Engine")
    assert 'data-tip="When the task completed, the agent actually returned real content' in html
    assert 'tabindex="0"' in html


def test_render_no_skills_tested_shows_placeholder_not_a_crash():
    result = {"target": "https://example.com/agent", "agent_name": "X", "skills_tested": [], "overall_pass": True}
    html = render(result, engine_label="Test Engine")
    assert "No skills were tested" in html
    checker = _WellFormedChecker()
    checker.feed(html)
    assert checker.errors == []
