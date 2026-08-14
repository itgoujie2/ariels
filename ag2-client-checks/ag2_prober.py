"""Drives one skill through AG2's real, official, first-party A2A client
(`autogen.a2a.client.A2aRemoteAgent`, the `ag2[a2a]` extra) to a stopping
point, at the same two-tier depth as the ADK and raw-SDK layers -- see
`PLAN.md` §11a for why this whole matrix of layers exists.

Unlike LangGraph, AG2 genuinely ships a first-party A2A client (per
`PLAN.md` §7's canonical-source table) -- confirmed live via package
inspection: `ag2[a2a]` pins `a2a-sdk<2,>=1.1.0`, the *current* SDK
generation (unlike ADK's `[a2a]` extra, which pins the *older* 0.3.x).

Three findings confirmed live before writing any code:
- `A2aRemoteAgent.a_generate_remote_reply()` internally loops
  (`while True`) whenever the remote task returns `input-required`,
  calling `self.a_get_human_input(prompt=...)` each iteration and
  feeding the answer back as the next message. Its default
  implementation blocks on stdin (a real console `input()` call) --
  AG2's own docstring says "Override this method to customize," which
  is exactly what a real automated integration has to do; this prober
  does the same.
- That internal loop has **no built-in iteration limit at all** --
  confirmed live against insideout.luthersystems.com (which naturally
  cycles through input-required due to its own real turn-2 backend bug):
  observed 7+ consecutive human-input round trips with no sign of
  stopping. The only sanctioned way to break out is answering "exit"
  (a hardcoded check in AG2's own source), which is what `MAX_TURNS` is
  enforced through here -- a real caller using AG2's own documented
  override pattern without adding their own turn limit could hang
  indefinitely against either a buggy backend or a legitimately slow-
  to-converge conversation.
- `_is_task_completed()` raises `A2aClientError` directly for
  `failed`/`rejected` states (cleaner than the other two clients, which
  surface those as plain normalized strings) -- caught here and reported
  as this skill's attributed error, consistent with the other layers'
  "loud failure, not a crash" discipline.
- AG2 chooses streaming vs. polling transport automatically based on
  the *card's own declared* `capabilities.streaming` -- a third,
  distinct default-selection strategy across the three native-client
  layers (raw SDK always defaults to streaming regardless of the card;
  ADK always defaults to non-streaming regardless of the card; AG2
  actually reads the card first).

Real limitation of AG2's own abstraction, documented rather than worked
around: `a_generate_remote_reply()`'s return value only distinguishes
"got real content back" from "didn't" -- it doesn't expose the
underlying Task's precise terminal state (`completed` vs. `canceled`
both return content the same way) the way the other two layers' raw
response dicts do. `final_state` here is therefore a best-effort label,
not a literal wire-state string.

A fourth finding, found live against insideout.luthersystems.com:
AG2's `response_message_from_a2a_task()` (in `autogen/a2a/utils.py`)
extracts an input-required task's clarifying question from
`task.history[-1]` -- but real agents (insideout included, per this
same session's own earlier findings with the other two clients) put
that question in `status.message` instead, not a separate `history`
array. When `history` doesn't carry it, AG2 silently falls back to a
generic `"Please provide input:"` placeholder, discarding the agent's
actual question entirely -- a real caller (or an LLM generating a
follow-up, as this prober does) has no way to know what was actually
asked. This is the same class of shape-normalization gap this whole
project has repeatedly found and fixed in `a2a_wire.py` -- except this
time in a third-party framework's own official client, observed rather
than fixed (consistent with this layer's whole purpose: test the real
client's real behavior, not paper over its gaps).
"""

from __future__ import annotations

from autogen.a2a.client import A2aRemoteAgent

from app import llm
from app.a2a_wire import explain_error

MAX_TURNS = 4


class _BoundedAutoReplyAgent(A2aRemoteAgent):
    """Overrides AG2's own documented extension point
    (`a_get_human_input`) to answer input-required turns automatically
    with an LLM-generated follow-up, instead of blocking on stdin --
    and enforces MAX_TURNS via AG2's own "exit" escape hatch, since the
    underlying loop has no bound of its own."""

    def _init_probe_state(self, skill: dict) -> None:
        self._probe_turn = 0
        self._probe_skill = skill
        self._probe_turns_log: list[dict] = []
        self._probe_exhausted = False

    async def a_get_human_input(self, prompt: str, **kwargs) -> str:
        question = prompt.split("\n", 1)[1] if "\n" in prompt else prompt
        self._probe_turn += 1
        if self._probe_turn > MAX_TURNS:
            self._probe_exhausted = True
            return "exit"
        followup = await llm.generate_followup(question, self._probe_skill)
        self._probe_turns_log.append({"sent": followup, "state": "input-required", "received": question})
        return followup


async def probe_skill_ag2(base_url: str, skill: dict) -> dict:
    """Returns a transcript dict shaped like the other layers' (turn,
    states, messages, turns_log, artifacts, final_state, final_message,
    error) so `app.assertions.check_transcript()` applies with only the
    caveat noted in this module's docstring (final_state is best-effort)."""
    probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)

    agent = _BoundedAutoReplyAgent(url=base_url, name="ag2_probe")
    agent._init_probe_state(skill)

    try:
        ok, reply = await agent.a_generate_remote_reply(messages=[{"role": "user", "content": probe_text}])
    except Exception as e:
        return {
            "turn": agent._probe_turn, "states": [], "messages": [], "turns_log": agent._probe_turns_log,
            "artifacts": [], "final_state": None, "final_message": None,
            "error": explain_error(f"{type(e).__name__}: {e}"),
        }

    turns_log = list(agent._probe_turns_log)
    states = [t["state"] for t in turns_log]

    if agent._probe_exhausted:
        return {
            "turn": agent._probe_turn, "states": states, "messages": [], "turns_log": turns_log,
            "artifacts": [], "final_state": "input-required", "final_message": None,
            "error": f"did not reach a terminal state within {MAX_TURNS} turns",
        }

    if not ok or reply is None:
        return {
            "turn": agent._probe_turn, "states": states, "messages": [], "turns_log": turns_log,
            "artifacts": [], "final_state": None, "final_message": None,
            "error": (
                "AG2 client returned no reply -- ambiguous outcome (its own "
                "a_generate_remote_reply() API doesn't distinguish this case "
                "from a genuine failure; see module docstring)"
            ),
        }

    text = reply.get("content", "") if isinstance(reply, dict) else str(reply)
    turns_log.append({"sent": probe_text if not turns_log else "(continued)", "state": "completed", "received": text})
    states.append("completed")

    return {
        "turn": agent._probe_turn + 1, "states": states, "messages": [text], "turns_log": turns_log,
        "artifacts": [text] if text else [], "final_state": "completed", "final_message": text,
        "error": None,
    }
