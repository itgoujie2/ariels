"""Drives one skill through CrewAI's own first-party A2A client
(`crewai.a2a.utils.delegation.aexecute_a2a_delegation`) to a stopping
point, at parity with the golden probers' and the other native-client
layers' per-skill test depth.

CrewAI (`crewai[a2a]`) ships a genuinely substantial, first-party A2A
integration -- this directly contradicts PLAN.md's stale §7 note ("no
official sample, only a third-party community adapter"), corrected as
part of building this layer. It pins `a2a-sdk~=0.3.10`, the same older
SDK generation as ADK's extra.

Key design decisions, confirmed live before building (not guessed):

- `aexecute_a2a_delegation()` returns a `TaskStateResult` TypedDict with a
  real `TaskState` enum in `status` -- the richest single status signal
  of any of the four clients tested so far (raw string states elsewhere).
- Card discovery only tries the current spec path
  (`/.well-known/agent-card.json`), no legacy fallback -- confirmed live
  against `p0stman.com` (legacy-card-only agent): a clean
  `httpx.HTTPStatusError: 404`, the same class of gap `a2a_wire.py`'s own
  `fetch_card()` already fixed for itself with `CARD_PATHS`.
- Same older-SDK card-model rigidity ADK's layer already found: confirmed
  live against `api.moltrust.ch`, a clean
  `pydantic.ValidationError: url Field required`.
- **Real finding, confirmed live against `insideout.luthersystems.com`**:
  when the task lands in `TaskState.input_required`, the actual
  clarifying-question text comes back in the **`error` field**, not
  `result` (which is `None`) or anywhere else. `result` is only populated
  for a genuinely completed task. A naive caller checking only `error` to
  decide pass/fail would misclassify a normal mid-conversation question as
  a failure. This prober reads `result.get("result") or result.get("error")
  or ""` uniformly for every state rather than special-casing
  `input_required` -- this also turns out to be the right read for a
  genuine `failed` state's own explanation text (see the continuation
  finding below), since CrewAI apparently multiplexes any non-`completed`
  explanatory text through `error` regardless of what put the task there.
- **Second real finding, confirmed live against the same agent**:
  continuing a multi-turn conversation requires passing back the *prior
  turn's* `task_id` (extracted from `result["history"][-1].task_id`) --
  passing only `context_id` silently starts a fresh task server-side (the
  agent re-asks its generic opening question, having forgotten turn 1
  entirely) rather than erroring, which is its own footgun. But passing
  the real `task_id` back, exactly as the API shape implies it should be
  used, hits a genuine live server error instead:
  `A2AClientJSONRPCError: -32603 ... "a2a task get: secret required"` --
  confirmed reproducible on a fresh run (different task_id each time, same
  error). The other three native clients tested this session (raw-sdk,
  ADK, AG2) all successfully complete multi-turn continuation against
  this exact same agent, so this is not "the agent doesn't support
  continuation" -- it is specific to how CrewAI's client resumes a task.
  `aexecute_a2a_delegation` catches this internally and returns a normal
  `TaskStateResult` with `status=TaskState.failed` (it does not raise) --
  this prober deliberately does *not* force that into a transcript-level
  protocol error the way a raised exception is; it flows through as an
  ordinary terminal `failed` state so `assertions.py`'s normal
  `explanation_present_on_failure` check evaluates it on its own merits
  (and passes, since the bug's own error text is a legitimate non-empty
  explanation) -- consistent with this project's zero-content-judgment
  discipline (PLAN.md §5.2): this layer only checks "did we reach a valid
  stopping state with content," not whether that state is a *good*
  outcome.
- **Third real finding, from the full 153-agent batch run**: CrewAI's own
  internal event-stream handling has a hardcoded stack-depth limit and
  crashes past it -- confirmed live against `43.129.31.185:8082` ("TiEQi",
  a real agent declaring 120 skills): `StackDepthExceededError: Event
  stack depth limit (100) exceeded. This usually indicates missing ending
  events.` A large/verbose real agent can trip an internal resource limit
  in CrewAI's own client that has nothing to do with the wire protocol
  itself.
- **Fourth real finding, from the same batch run**: CrewAI's card-fetch
  step is stricter than necessary about a card's own declared
  `security`/`securitySchemes` -- confirmed live against
  `agent.co-legal.be` (one of this project's 10 ground-truth
  `fully_working` agents, reachable with zero auth by every other client
  and by our own hand-rolled prober): `A2AClientHTTPError: HTTP Error
  401: AgentCard requires authentication but no auth scheme provided`.
  CrewAI refuses to proceed without a configured auth scheme purely
  because the card *declares* one is supported, even though the agent
  doesn't actually enforce it for this call -- a real, distinct
  reachability cost specific to this client, not the agent's own
  behavior.
"""

from __future__ import annotations

from crewai.a2a.utils.delegation import aexecute_a2a_delegation

from app import llm
from app.a2a_wire import explain_error

MAX_TURNS = 4
CONTINUABLE_STATES = {"input-required"}

# Layer-specific explanation for this module's own documented finding
# (see module docstring): CrewAI's client only tries the current A2A
# spec's well-known path (/.well-known/agent-card.json), no fallback to
# the older, legacy path (/.well-known/agent.json) some real agents
# still use -- confirmed live against p0stman.com, whose card is only
# served at that legacy path. The raw httpx.HTTPStatusError text always
# embeds the failing URL, so a card-discovery 404 reliably contains
# "well-known" -- distinct from an unrelated 404 hit elsewhere (e.g.
# posting to a wrong RPC endpoint), which wouldn't mention that path at
# all. Safe to state definitively that the card exists elsewhere: by
# the time probe_skill_crewai ever runs, run_parity.py's own main() has
# already fetched this exact agent's card successfully via
# a2a_wire.fetch_card() (that's how its skills were enumerated for this
# report) -- and since fetch_card() tries the *current* path first, the
# same one CrewAI's client also tries, CrewAI failing there while
# fetch_card() succeeded overall means the card can only be reachable
# via the *other* (legacy) path fetch_card() falls back to.
_LAYER_ERROR_EXPLANATIONS = [
    (
        "well-known",
        "CrewAI's client only tries the current A2A spec's well-known path "
        "(/.well-known/agent-card.json) -- it has no fallback to the older, legacy path "
        "(/.well-known/agent.json) that some real agents (like p0stman.com) still use "
        "exclusively. This project's own broader discovery already found this agent's card "
        "successfully (that's how its skills were enumerated for this report), which proves "
        "the card exists -- this is a limitation of CrewAI's own client, not a genuinely "
        "undiscoverable agent.",
    ),
]


def _explain_layer_error(raw: str) -> str:
    for substring, explanation in _LAYER_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    return explain_error(raw)


def _state_str(status) -> str:
    """TaskState enum -> our lowercase-kebab convention (`_normalize()`
    elsewhere already produces this shape for the other three clients,
    so downstream `assertions.py` checks apply unchanged)."""
    raw = status.value if hasattr(status, "value") else str(status)
    return raw.replace("_", "-")


async def probe_skill_crewai(base_url: str, skill: dict) -> dict:
    """Returns a transcript dict shaped exactly like the golden probers'
    and the other native-client layers', so
    `app.assertions.check_transcript()` applies unchanged."""
    probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)

    turn = 0
    states: list[str] = []
    messages: list[str] = []
    turns_log: list[dict] = []
    artifacts: list[str] = []
    error: str | None = None
    final_state: str | None = None
    final_message: str | None = None
    context_id: str | None = None
    task_id: str | None = None
    history = None

    next_text = probe_text
    sent_desc = probe_text

    while turn < MAX_TURNS:
        try:
            result = await aexecute_a2a_delegation(
                endpoint=base_url,
                auth=None,
                timeout=30,
                task_description=next_text,
                context_id=context_id,
                task_id=task_id,
                conversation_history=history,
            )
        except Exception as e:
            error = _explain_layer_error(f"turn {turn + 1}: {type(e).__name__}: {e}")
            break
        turn += 1

        status = result.get("status")
        state = _state_str(status) if status is not None else "unknown"
        # Real finding (see module docstring): `result` only carries text on
        # a genuine `completed` task; `error` carries it for every other
        # non-empty state CrewAI returns (`input_required`'s clarifying
        # question, but also a `failed` task's own explanation) -- treated
        # uniformly here rather than special-cased per state, so a `failed`
        # task still flows through to assertions.py's normal
        # `explanation_present_on_failure` check instead of being forced
        # into a hard protocol-level error.
        text = result.get("result") or result.get("error") or ""

        states.append(state)
        messages.append(text)
        turns_log.append({"sent": sent_desc, "state": state, "received": text})
        final_state = state
        final_message = text
        history = result.get("history")
        if history:
            last = history[-1]
            context_id = last.context_id or context_id
            task_id = last.task_id or task_id

        if state in CONTINUABLE_STATES:
            if turn >= MAX_TURNS:
                error = f"did not reach a terminal state within {MAX_TURNS} turns"
                break
            followup_text = await llm.generate_followup(text, skill)
            next_text = followup_text
            sent_desc = followup_text
            continue
        else:
            break

    return {
        "turn": turn, "states": states, "messages": messages, "turns_log": turns_log,
        "artifacts": artifacts, "final_state": final_state, "final_message": final_message,
        "error": error,
    }
