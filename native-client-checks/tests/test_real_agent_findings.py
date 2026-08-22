"""Regression tests for specific real-world conformance gaps discovered
while probing actual live agents -- distinct from test_a2a_wire.py, which
only checks that _normalize() can *parse* a given shape. These tests check
the full pipeline (normalize -> transcript -> assertions) against the exact
real response that caused a real failure, to lock in that our conformance
checks keep catching it.

Rule: every time probing a real agent turns up an issue, add
it here -- not just new parseable shapes.
"""

import asyncio

from app.a2a_wire import _normalize, resolve_target
from app.assertions import check_transcript


def _norm(*args, **kwargs):
    """_normalize() is async (it may fall back to an LLM call for genuinely
    unrecognized content shapes) -- none of the fixtures here trigger that
    path, so this just bridges sync test functions to the async call."""
    return asyncio.run(_normalize(*args, **kwargs))


def test_moltrust_boilerplate_reply_parses_and_has_content(load_fixture):
    """api.moltrust.ch (real, live, production v1.0 agent) returns the exact
    same static self-description for every one of its 5 declared skills,
    regardless of what's asked or whether the caller authenticates --
    confirmed with a freshly-registered valid DID. Its skill logic is
    genuinely real and working (confirmed separately via its REST API), but
    its A2A layer never routes to it.

    IMPORTANT — what this test does NOT claim: `content_present_on_completion`
    correctly PASSES this transcript, because there genuinely is content
    (the boilerplate text itself is non-empty). "The same reply regardless
    of the question" is a content-quality/repetition defect, not a missing-
    content one -- per assertions.py's own zero-content-judgment discipline,
    a single transcript in isolation cannot distinguish a repeated boilerplate
    reply from a real, correct, differentiated one. Catching this specific
    defect deterministically would need a *cross-skill* check (do N distinct
    probes yield the exact same response text?) that doesn't exist yet.
    This test just locks in that the transcript still parses
    and passes the mechanical checks, so a future contributor doesn't
    mistake that for "this agent is fine."
    """
    result = load_fixture("v10_moltrust_boilerplate_reply")
    normalized = _norm("1.0", result, None, None)
    assert normalized["state"] == "completed"
    assert normalized["artifacts"] == []  # no artifact -- this is a bare Message result, not a Task
    assert normalized["agent_text"] != ""  # content exists -- just identical every time (untested here)

    transcript = {
        "final_state": normalized["state"],
        "turn": 1,
        "artifacts": normalized["artifacts"],
        "final_message": normalized["agent_text"],
    }
    checks = check_transcript(transcript)
    content_check = next(c for c in checks if c["check"] == "content_present_on_completion")
    assert content_check["passed"] is True


def test_colegal_message_only_reply_passes_content_check(load_fixture):
    """agent.co-legal.be (real, live, production Belgian legal Q&A agent)
    gave a genuinely correct, distinct, substantive answer to every one of 9
    declared skills -- but never populates `artifacts`, only status.message.
    Before content_present_on_completion (formerly artifact_present_on_
    completion, artifacts-only), this was a false negative: a fully working
    agent reported as failing on every single skill. Also exercised a real
    resolve_target() bug on the way here -- see
    test_resolve_target_prefers_supported_interfaces_over_flat_url in
    test_a2a_wire.py."""
    result = load_fixture("v10_colegal_message_only_reply")
    normalized = _norm("1.0", result, None, None)
    assert normalized["state"] == "completed"
    assert normalized["artifacts"] == []
    assert normalized["agent_text"] != ""

    transcript = {
        "final_state": normalized["state"],
        "turn": 1,
        "artifacts": normalized["artifacts"],
        "final_message": normalized["agent_text"],
    }
    checks = check_transcript(transcript)
    content_check = next(c for c in checks if c["check"] == "content_present_on_completion")
    assert content_check["passed"] is True


def test_resolve_target_colegal_card_shape(load_fixture):
    """The exact real card shape that caused the dialect-misdetection bug:
    both a flat `url` and a `supportedInterfaces` entry, no top-level
    protocolVersion. Full regression-test coverage for this specific shape
    also lives in test_a2a_wire.py's dedicated resolve_target tests; this
    one confirms it against the actual captured card, not just a
    hand-built minimal one."""
    card = {
        "name": "colegal-public-assistant",
        "url": "https://agent.co-legal.be/a2a/jsonrpc",
        "supportedInterfaces": [
            {
                "url": "https://agent.co-legal.be/a2a/jsonrpc",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ],
    }
    rpc_url, dialect = resolve_target(card, "https://agent.co-legal.be")
    assert dialect == "1.0"


def test_paki_v03_flat_shape_despite_v10_request(load_fixture):
    """paki-api.elfresonero.workers.dev (real, live, v1.0-declaring art
    agent) replies in pure v0.3-flat shape (kind:'task' at top level,
    lowercase state) even when called with the v1.0 method/header -- the
    mirror-image mismatch from agent.co-legal.be (which always replies
    v1.0-style regardless of request dialect). Before _normalize became
    shape-driven (checking the response's own structure via
    _task_from_result/_message_from_result instead of trusting the
    `dialect` parameter to predict it), this raised
    UnrecognizedResponseShapeError on every one of this agent's skills."""
    result = load_fixture("v03_flat_shape_from_v10_request")
    normalized = _norm("1.0", result, None, None)  # requested as 1.0, shaped as 0.3
    assert normalized["state"] == "completed"
    assert normalized["task_id"] == "47dd55be-f370-4b41-bfab-1cc062f6a6ec"
    assert "Gracias" in normalized["agent_text"]

    transcript = {
        "final_state": normalized["state"],
        "turn": 1,
        "artifacts": normalized["artifacts"],
        "final_message": normalized["agent_text"],
    }
    checks = check_transcript(transcript)
    content_check = next(c for c in checks if c["check"] == "content_present_on_completion")
    assert content_check["passed"] is True


def test_agent_ready_data_only_reply_has_content(load_fixture):
    """agent-ready.dev's public 'ask' skill (real, live, no API key needed)
    gave a genuinely rich, real answer -- Schema.org-typed search results --
    entirely as a `kind: "data"` part, no `kind: "text"` part at all. Before
    _extract_texts() handled data parts, this was a false negative: a fully
    substantive real answer reported as empty content, identical in kind
    (if not mechanism) to the colegal and 2s.io false negatives."""
    result = load_fixture("v03_data_only_reply")
    normalized = _norm("0.3", result, None, None)
    assert normalized["state"] == "completed"
    assert normalized["agent_text"] != ""
    assert "llms.txt" in normalized["agent_text"]

    transcript = {
        "final_state": normalized["state"],
        "turn": 1,
        "artifacts": normalized["artifacts"],
        "final_message": normalized["agent_text"],
    }
    checks = check_transcript(transcript)
    content_check = next(c for c in checks if c["check"] == "content_present_on_completion")
    assert content_check["passed"] is True


def test_zee_task_with_no_kind_discriminator(load_fixture):
    """p0stman.com/Zee (real, live agent, discovered via the legacy
    /.well-known/agent.json path -- see test_fetch_card_falls_back_to_legacy_
    well_known_path in test_a2a_wire.py for that half of this finding) omits
    the "kind" discriminator on its Task object entirely -- just
    id/contextId/status/artifacts, no "kind":"task" anywhere. Before
    _task_from_result() accepted "status present, no kind" as sufficient
    Task evidence, this raised UnrecognizedResponseShapeError on every
    skill. Also uses {"type":"text",...} parts instead of {"kind":"text",...}
    -- already handled fine, since _extract_texts only checks for a "text"
    key, agnostic to the discriminator field name."""
    result = load_fixture("v03_task_no_kind_discriminator")
    normalized = _norm("1.0", result, None, None)  # requested as 1.0, shaped with no discriminator at all
    assert normalized["state"] == "completed"
    assert "p0stman" in normalized["agent_text"]

    transcript = {
        "final_state": normalized["state"],
        "turn": 1,
        "artifacts": normalized["artifacts"],
        "final_message": normalized["agent_text"],
    }
    checks = check_transcript(transcript)
    content_check = next(c for c in checks if c["check"] == "content_present_on_completion")
    assert content_check["passed"] is True


def test_kunlun_status_as_plain_string_does_not_crash(load_fixture):
    """ai.syln.cn (real, live, "昆仑/Kunlun" agent) puts the task state
    directly as a plain string (`"status": "completed"`), not nested under
    a state field like every other agent tested. Before _normalize()
    normalized this, calling status.get("state", ...) on a raw string
    crashed with "'str' object has no attribute 'get'" -- a real Python
    exception leaking out of our own parsing code (surfaced as
    reason: "other" in version_probe, not a clean protocol-level error).
    Same agent also puts the reply message as a sibling of "status"
    (task.message) rather than nested inside it (status.message);
    confirms that's handled too, not just the crash."""
    result = load_fixture("v03_status_as_plain_string")
    normalized = _norm("0.3", result, None, None)
    assert normalized["state"] == "completed"
    assert "昆仑" in normalized["agent_text"]

    transcript = {
        "final_state": normalized["state"],
        "turn": 1,
        "artifacts": normalized["artifacts"],
        "final_message": normalized["agent_text"],
    }
    checks = check_transcript(transcript)
    content_check = next(c for c in checks if c["check"] == "content_present_on_completion")
    assert content_check["passed"] is True


def test_rettfrabonden_artifacts_sibling_of_task(load_fixture):
    """rettfrabonden.com (real, live) puts "artifacts" as a sibling of the
    nested "task" wrapper rather than inside it -- {result: {task: {...},
    artifacts: [...]}}. Before _task_from_result() merged the sibling
    artifacts in, deterministic extraction found nothing here despite real
    content being one level up, falling through to the LLM fallback
    unnecessarily for a shape that's cheaply, deterministically handleable."""
    result = load_fixture("v10_artifacts_sibling_of_task")
    normalized = _norm("1.0", result, None, None)
    assert normalized["state"] == "completed"
    assert len(normalized["artifacts"]) == 1
    assert "Homme" in normalized["agent_text"]

    transcript = {
        "final_state": normalized["state"],
        "turn": 1,
        "artifacts": normalized["artifacts"],
        "final_message": normalized["agent_text"],
    }
    checks = check_transcript(transcript)
    content_check = next(c for c in checks if c["check"] == "content_present_on_completion")
    assert content_check["passed"] is True


def test_insideout_multiturn_continuation_backend_error_reported_cleanly():
    """insideout.luthersystems.com (real, live "InsideOut" cloud-architect
    agent): its first response is completely legitimate -- a genuine,
    substantive `input-required` reply asking a real clarifying question
    about the requested architecture. But the *second* turn (continuing the
    same task via taskId/contextId, exactly as the spec expects) fails with
    `A2A error -32603: internal: internal: a2a task get: secret required` --
    a real bug in their own task-resumption backend (an internal secret/
    credential their task-lookup service needs is missing or misconfigured),
    not anything missing from our request.

    This is a genuinely new failure mode this session: not a shape-parsing
    gap and not a version/dialect mismatch, but a live multi-turn
    conversation breaking mid-flight on the *provider's* side. No code
    change was needed -- prober.py's send_followup already catches this
    exception and reports it as this skill's attributed error rather than
    crashing the run -- so this test just locks in that behavior stays
    correct for this exact real error text."""
    from app.assertions import check_transcript

    transcript = {
        "final_state": "input-required",
        "turn": 1,
        "artifacts": [],
        "final_message": "Here is a list of features I think your 3-tier web application needs...",
        "error": "RuntimeError: A2A error -32603: internal: internal: a2a task get: secret required",
    }
    checks = check_transcript(transcript)
    assert len(checks) == 1
    stopping_check = checks[0]
    assert stopping_check["check"] == "reached_stopping_point"
    assert stopping_check["passed"] is False
    assert "secret required" in stopping_check["detail"]


def test_insideout_design_deploy_cloud_skill_hits_bespoke_rate_limit_error():
    """insideout.luthersystems.com (real, live, via product/'s orchestrator):
    a *second*, distinct backend quirk on this same agent, found on its
    `design-deploy-cloud` skill specifically -- `A2A error -32004: rate
    limit exceeded — try again in a moment: this operation is not
    supported`. Unlike the -32603 "secret required" bug above (a
    multi-turn continuation failure), this is a first-turn error with a
    non-standard JSON-RPC code (-32004) and an oddly self-contradictory
    message that conflates a rate-limit condition with "not supported" --
    reads like a bespoke gate on this specific skill on their backend,
    not anything wrong with our request. No code change needed --
    check_transcript() already reports any first-turn error as a clean,
    attributed reached_stopping_point failure; this test locks in that
    this exact real error text keeps being surfaced that way rather than
    crashing or being silently misclassified."""
    from app.assertions import check_transcript

    transcript = {
        "final_state": None,
        "turn": 1,
        "artifacts": [],
        "final_message": None,
        "error": "RuntimeError: A2A error -32004: rate limit exceeded — try again in a moment: this operation is not supported",
    }
    checks = check_transcript(transcript)
    assert len(checks) == 1
    stopping_check = checks[0]
    assert stopping_check["check"] == "reached_stopping_point"
    assert stopping_check["passed"] is False
    assert "rate limit exceeded" in stopping_check["detail"]
    assert "not supported" in stopping_check["detail"]


def test_ambr_underscore_state_normalized_and_continuable(load_fixture):
    """getamber.dev ("Ambr", real, live) uses "input_required" (underscore)
    instead of the spec's "input-required" (hyphen) -- the first real agent
    all session to naturally need a multi-turn input-required resolution.
    Before state normalization stripped underscores, this was invisible to
    both assertions.py's known-state checks AND, more importantly,
    prober.py's CONTINUABLE_STATES check -- meaning the prober would never
    even attempt a follow-up for a real agent that genuinely needed one."""
    from native_client_prober import CONTINUABLE_STATES

    result = load_fixture("v03_state_with_underscore")
    normalized = _norm("0.3", result, None, None)
    assert normalized["state"] == "input-required"
    assert normalized["state"] in CONTINUABLE_STATES

    transcript = {
        "final_state": normalized["state"],
        "turn": 1,
        "artifacts": normalized["artifacts"],
        "final_message": normalized["agent_text"],
    }
    checks = check_transcript(transcript)
    assert all(c["passed"] for c in checks)
