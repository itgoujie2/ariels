"""Single-engine standalone HTML report -- renders exactly one prober's
own `result` dict (the shape `run_prober.py`'s/`run_parity.py`'s own
`main()` already produces: `target`, `agent_name`, `declared_skills`,
`skills_tested`, `overall_pass`, `error` on top-level failure) into one
self-contained HTML page. No server, no account, no product/ dependency
-- exactly what a standalone open-source install can produce on its own.

Adapted from `product/app/report/render_detail.py`'s per-skill rendering
(`_render_transcript_turn`/`_render_checks`/`_render_skill_card`) --
copied, not imported, since `product/` is a separate, private layer this
open package has no dependency on (same "copy, don't import" convention
`render_detail.py`/`theme.py` themselves already follow relative to
`reports/`). `THEME_CSS` below is a trimmed copy of
`product/app/report/theme.py`'s own variable block (same colors/layout)
-- trimmed to the pieces a single-engine page actually uses (masthead,
skill cards, checks, transcript turns), and skipping `theme.py`'s
embedded-webfont dependency (it reads a file from `reports/`, which
stays private) in favor of a plain system font stack, so this package
has zero file dependency outside its own tree.

Deliberately does NOT render a cross-engine comparison or AI-Judgment
section -- those need every one of the other engines' own results side
by side, plus account/entitlement plumbing that only exists in the
hosted product. This is exactly what one engine's own CLI can produce
running completely standalone.
"""
from __future__ import annotations

import html as _html
from typing import Any

# Mirrors assertions.py's TERMINAL_STATES minus "completed" -- a
# terminal state that isn't a clean success.
FAILURE_STATES = {"failed", "canceled", "rejected"}


def _esc(s: Any) -> str:
    return _html.escape(str(s)) if s is not None else ""


def _check_dot_color(passed: bool) -> str:
    return "var(--good)" if passed else "var(--bad)"


def _render_transcript_turn(turn: dict) -> str:
    sent = _esc(turn.get("sent"))
    received = _esc(turn.get("received"))
    state = _esc(turn.get("state"))
    return f"""<div class="turn">
      <div class="role">Sent</div>
      <div class="sent">{sent}</div>
      <div class="role">Received <span class="state">[{state}]</span></div>
      <div class="received">{received}</div>
    </div>"""


# Plain-language explanation per check name, for a hover tooltip -- these
# are internal check identifiers (from assertions.py's own
# check_transcript()), not written for a reader unfamiliar with this
# project's own vocabulary.
_CHECK_EXPLANATIONS = {
    "reached_stopping_point": "The skill reached a real stopping point (finished, blocked on more info, or failed) instead of erroring out or hanging.",
    "known_task_state": "The final state the agent reported is one this project recognizes, not an unexpected value.",
    "reached_terminal_or_explainable_block": "The skill ended in a genuine completion/failure, or a legitimate ‘needs more input’ block — not stuck in an unrecognized state.",
    "content_present_on_completion": "When the task completed, the agent actually returned real content (an artifact or reply text), not an empty response.",
    "explanation_present_on_failure": "When the task failed, the agent explained why, rather than failing silently.",
}


def _render_checks(checks: list[dict]) -> str:
    if not checks:
        return ""
    items = ""
    for c in checks:
        passed = bool(c.get("passed"))
        dot_color = _check_dot_color(passed)
        check_id = c.get("check")
        name = _esc(check_id)
        detail = _esc(c.get("detail"))
        explanation = _CHECK_EXPLANATIONS.get(check_id)
        # CSS-only tooltip (data-tip + ::after in THEME_CSS below), not the
        # native `title` attribute -- confirmed live elsewhere in this
        # project that native tooltips are unreliable (see
        # render_detail.py's own comment on this exact finding).
        tip_attr = f' data-tip="{_esc(explanation)}" tabindex="0"' if explanation else ""
        items += f"""<div class="check">
      <span class="dot" style="background:{dot_color}"></span>
      <span class="name"{tip_attr}>{name}</span>
      <span class="detail">{detail}</span>
    </div>"""
    return f'<div class="checks">{items}</div>'


def _render_skill_card(skill: dict, idx: int) -> str:
    skill_id = _esc(skill.get("skill_id"))
    passed = bool(skill.get("pass"))
    badge_class = "good" if passed else "bad"
    badge_label = "pass" if passed else "fail"
    elapsed = skill.get("elapsed_ms")
    elapsed_txt = f"{elapsed:.0f}ms" if isinstance(elapsed, (int, float)) else "&mdash;"
    raw_final_state = skill.get("final_state")
    final_state = _esc(raw_final_state)
    turns = skill.get("transcript") or []
    turns_html = "".join(_render_transcript_turn(t) for t in turns)
    checks = skill.get("checks") or []
    checks_html = _render_checks(checks)

    # A skill can show a legitimate exchange in its transcript AND an
    # error that looks unrelated -- it isn't: the error is from a LATER
    # exchange attempt that raised before producing any response (never
    # added to the transcript), not about the turn(s) shown above.
    later_failure_note = ""
    stopping_check = next((c for c in checks if c.get("check") == "reached_stopping_point"), None)
    if stopping_check and not stopping_check.get("passed") and turns:
        later_failure_note = (
            f'<div class="notice"><strong>A later exchange failed:</strong> {_esc(stopping_check.get("detail"))} '
            "&mdash; not shown above, since no response was ever recorded for it. The conversation above is unrelated "
            "and completed successfully as far as it went.</div>"
        )

    # A failed/canceled/rejected task's own "Received" text is the
    # agent's own failure explanation, not a normal answer -- flagged
    # structurally (by state), not by pattern-matching the text itself.
    failure_state_note = ""
    if raw_final_state in FAILURE_STATES and turns:
        failure_state_note = (
            f'<p class="failure-caption">This task ended in a <code>{final_state}</code> state &mdash; the "Received" '
            "text above is the agent's own failure explanation, not a normal answer. A passing check here means the "
            "agent explained why it failed, not that it succeeded.</p>"
        )

    open_attr = " open" if idx == 0 else ""
    return f"""<details class="card"{open_attr}>
    <summary class="card-head">
      <span class="title"><span class="caret">&#9656;</span><span class="skill-eyebrow">Skill</span>{skill_id}</span>
      <span class="card-meta">state: {final_state} &middot; {elapsed_txt}</span>
      <span class="badge {badge_class}">{badge_label}</span>
    </summary>
    <div class="card-body">
      {checks_html}
      {turns_html}
      {failure_state_note}
      {later_failure_note}
    </div>
  </details>"""


THEME_CSS = """
:root {
  --bg: #f2f4f4;
  --surface: #ffffff;
  --surface-2: #e9edee;
  --ink: #14191b;
  --muted: #5b6b6f;
  --muted-strong: #8a9599;
  --accent: #b3572a;
  --accent-soft: rgba(179,87,42,0.12);
  --good: #2f7d4c;
  --bad: #a83f2e;
  --warn: #a67a1f;
  --line: rgba(20,25,27,0.12);
  --line-soft: rgba(20,25,27,0.07);
  --shadow: 0 1px 2px rgba(20,25,27,0.06), 0 8px 24px rgba(20,25,27,0.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1517;
    --surface: #161e21;
    --surface-2: #1c2529;
    --ink: #e7edee;
    --muted: #93a5aa;
    --muted-strong: #6d7d81;
    --accent: #e08a4f;
    --accent-soft: rgba(224,138,79,0.16);
    --good: #5aab73;
    --bad: #d1685a;
    --warn: #d9b03f;
    --line: rgba(255,255,255,0.10);
    --line-soft: rgba(255,255,255,0.06);
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
  }
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 900px; margin: 0 auto; padding: 0 28px; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
h1, h2, h3 { font-weight: 800; letter-spacing: 0.01em; text-wrap: balance; margin: 0; }

.masthead { border-bottom: 1px solid var(--line); padding: 22px 0; }
.masthead .wrap { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.brand { display: flex; align-items: baseline; gap: 12px; }
.brand .mark { font-weight: 800; font-size: 24px; letter-spacing: 0.02em; }
.brand .tagline { color: var(--muted); font-size: 14px; }
.meta { color: var(--muted); font-size: 13px; text-align: right; }
.meta .mono { color: var(--muted-strong); }

main.wrap { padding: 28px 28px 56px; }
.stat-row { display: flex; gap: 40px; flex-wrap: wrap; margin: 8px 0 28px; }
.stat { min-width: 140px; }
.stat .num { font-weight: 800; font-size: 32px; color: var(--accent); font-variant-numeric: tabular-nums; }
.stat .cap { margin-top: 4px; font-size: 12.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.notice { background: var(--accent-soft); border: 1px solid var(--line); border-radius: 6px; padding: 12px 16px; font-size: 14px; margin: 16px 0; }

.card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; margin-bottom: 12px; overflow: hidden; }
.card-head {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 14px 18px; cursor: pointer; user-select: none; list-style: none;
}
.card-head::-webkit-details-marker { display: none; }
.card-head .title { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 15px; }
.skill-eyebrow { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; color: var(--accent); border: 1px solid var(--accent); border-radius: 3px; padding: 2px 6px; }
.card-head .caret { color: var(--muted); transition: transform 0.15s ease; font-size: 12px; }
details[open] > .card-head .caret { transform: rotate(90deg); }
.badge { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; padding: 3px 9px; border-radius: 20px; white-space: nowrap; }
.badge.good { background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good); }
.badge.bad { background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); }
.card-meta { color: var(--muted); font-size: 12.5px; font-family: ui-monospace, monospace; }
.card-body { padding: 4px 18px 18px; border-top: 1px solid var(--line-soft); }

.turn { border-left: 3px solid var(--line); padding: 10px 0 10px 14px; margin: 10px 0; }
.turn .role { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 700; margin-bottom: 4px; }
.turn .sent { color: var(--ink); font-size: 14px; margin-bottom: 8px; white-space: pre-wrap; }
.turn .received { color: var(--ink); font-size: 14px; white-space: pre-wrap; background: var(--surface-2); border-radius: 4px; padding: 8px 10px; }
.turn .state { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted-strong); margin-top: 4px; }
.failure-caption { color: var(--warn); font-size: 13px; margin: 8px 0 0; font-style: italic; }

.checks { margin: 12px 0; display: flex; flex-direction: column; gap: 6px; }
.check { display: flex; align-items: baseline; gap: 8px; font-size: 13.5px; }
.check .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; margin-top: 4px; }
.check .name { font-weight: 600; }
.check .detail { color: var(--muted); }

.check .name[data-tip] { border-bottom: 1px dotted var(--muted-strong); cursor: help; position: relative; }
.check .name[data-tip]:hover::after, .check .name[data-tip]:focus::after {
  content: attr(data-tip);
  position: absolute; left: 0; bottom: calc(100% + 8px);
  background: var(--ink); color: var(--bg); font-weight: 400; font-size: 12.5px; line-height: 1.4;
  padding: 8px 10px; border-radius: 6px; width: max-content; max-width: 300px; white-space: normal;
  box-shadow: var(--shadow); z-index: 20;
}

footer { padding: 32px 0 56px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); margin-top: 32px; }
::selection { background: var(--accent-soft); }
"""


def render(result: dict, engine_label: str) -> str:
    """Renders one prober's own `result` dict (from its `main()`) into a
    complete, self-contained HTML page. `engine_label` is a short,
    human-readable name for the engine that produced it (e.g. "LangGraph
    golden prober", "CrewAI native client")."""
    target = _esc(result.get("target"))
    agent_name = _esc(result.get("agent_name")) or "(unnamed agent)"
    engine = _esc(engine_label)

    if result.get("error"):
        body = f'<div class="notice"><strong>Could not complete this check:</strong> {_esc(result["error"])}</div>'
    else:
        skills = result.get("skills_tested") or []
        passed_count = sum(1 for s in skills if s.get("pass"))
        stat_row = (
            '<div class="stat-row"><div class="stat">'
            f'<div class="num">{passed_count}/{len(skills)}</div><div class="cap">skills pass</div>'
            "</div></div>"
        )
        cards = "".join(_render_skill_card(s, i) for i, s in enumerate(skills)) or "<p>No skills were tested.</p>"
        body = stat_row + cards

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ariel report &mdash; {engine} &mdash; {target}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{THEME_CSS}</style>
</head>
<body>
<header class="masthead">
  <div class="wrap">
    <div class="brand"><span class="mark">ARIEL</span><span class="tagline">{engine}</span></div>
    <div class="meta">target: <span class="mono">{target}</span><br>agent: {agent_name}</div>
  </div>
</header>
<main class="wrap">
{body}
</main>
<footer class="wrap">Generated standalone by this open-source prober &mdash; no account or hosted service involved.</footer>
</body>
</html>"""
