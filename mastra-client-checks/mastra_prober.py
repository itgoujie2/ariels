"""Drives one skill through Mastra's own first-party A2A client
(`@mastra/core/a2a`'s `A2AAgent`) to a stopping point, at parity with
the golden probers' and the other native-client layers' per-skill test
depth -- but via a cross-language bridge, since Mastra is JS/TS, not
Python.

**Architecture note**: this is the first cross-language layer in the
native-client matrix (confirmed via a broader scan across languages,
not just Python -- see `PLAN.md` §11a). `mastra_bridge.mjs` is a
persistent Node.js subprocess (spawned once per skill, kept alive for
the whole multi-turn probe) that holds one real `A2AAgent` instance and
speaks newline-delimited JSON over stdin/stdout. This prober drives it
turn by turn exactly like every other layer -- generating follow-up
text via `app.llm.generate_followup()`, applying `MAX_TURNS`/
`CONTINUABLE_STATES` -- then feeds the raw `task`/`message` JSON the
bridge returns straight through `app.a2a_wire._normalize()`, since
`@a2a-js/sdk`'s wire-shape types are already the same camelCase/`kind`-
discriminated JSON `_normalize()` expects (confirmed live: no
conversion needed at all, unlike the protobuf/pydantic layers).

The bridge process must stay alive across turns (not one process per
turn) specifically because Mastra's own `resumeGenerate()` needs it:
its run state (`context_id`/`task_id`/`waitingForInput` keyed by
`runId`) lives in an in-memory `Map` internal to the `A2AAgent`
instance -- there is no way to reconstruct it from a fresh process.

Key findings, all confirmed live before/while building:

- Confirmed via direct package inspection that Mastra ships not one but
  *two* different A2A client surfaces: `MastraClient.getA2A(agentId)`
  (for calling an agent *hosted on your own Mastra deployment*, by ID)
  and the genuinely general-purpose one used here,
  `@mastra/core/a2a`'s `A2AAgent({url, headers})` -- built to consume
  *any* third-party A2A agent as a subagent, the right analog to every
  other layer's "point this at any agent" client.
- `A2AAgent.getAgentCard()`/`generate()` work cleanly standalone,
  without needing to wire the agent into a parent Mastra `Agent`/LLM
  pipeline at all -- confirmed live against `2s.io`: card fetch and a
  real reply both worked with nothing but `new A2AAgent({url})` and one
  `generate()` call.
- **Decisive finding, the most serious of this session alongside the
  Microsoft Agent Framework one**: `A2AAgent.resumeGenerate()` -- called
  exactly as its own types/API require (matching `runId`, a
  `resumePayload`-shaped `resumeData`) -- does **not actually continue
  the same task at all**. Confirmed live against
  `insideout.luthersystems.com`: turn 2's returned `task.id` *and*
  `task.contextId` both differ from turn 1's, and the reply is a
  generic re-ask of the agent's opening question, not a continuation of
  the specific conversation. Confirmed by reading the source
  (`@mastra/core/dist/a2a/index.js`): the continuation message is built
  with `referenceTaskIds: [state.taskId]` -- a *reference/linking*
  field per the A2A spec, not the actual continuation mechanism -- and
  never sets the outgoing message's own `taskId` field, which is what a
  server needs to recognize "this is the same task, resume it" rather
  than "this references an earlier task for context." The same footgun
  class already found in CrewAI's client, but here despite an API
  surface (`resumePayload`, `waitingForInput`, `resumeGenerate`) whose
  naming strongly implies genuine suspend/resume support, and even
  though `A2AAgentResumePayload` carries the actual `taskId` right there
  in a field the method never uses for that purpose.
- **The bug is more deceptive than it first looks**: a real multi-turn
  probe run through this exact prober against `insideout.luthersystems.com`
  produced a fully plausible, topically-coherent 4-turn conversation
  (feature list -> AWS/S3/CloudFront choice -> Terraform generation ->
  a specific CloudFront TTL adjustment, each reply correctly referencing
  specifics from the "previous" turn) -- which looks, on casual reading
  of the transcript, like successful continuation. It is not: a direct
  side-by-side check (calling the bridge's `generate` then `resume` with
  identical code, printing `task.id`/`task.contextId` each time) proves
  every turn gets a brand-new, unrelated task and context server-side.
  The illusion of continuity is produced entirely by this project's own
  follow-up-generation step (`app.llm.generate_followup()`, which reads
  the agent's *previous reply text* and writes a human-sounding
  response that naturally restates enough of it) -- a stateless
  single-turn agent on the other end can still sound coherent when
  handed a message that itself repeats the relevant context, with zero
  real session/task continuity underneath. A real caller relying on
  Mastra's own `resumeGenerate()` for anything that depends on genuine
  server-side task continuity (billing/metering per task, an
  authorization scope tied to the original task, artifacts or history
  attached to the first task) would silently lose all of it on every
  single turn, while a transcript-only observer would see nothing wrong.
- Run live at full scale (153 agents), two more real crashes turned up
  in the client's own code, both raw, unhandled JS exceptions rather
  than a clean, attributed A2A-level error: (1) `TypeError: Cannot read
  properties of undefined (reading 'message')` against three different
  real agents (`agent.co-legal.be`, `getamber.dev`, `postalform.com`) --
  an unconditional property access on something the client assumed
  would always be present; (2) `TypeError: Failed to parse URL from
  undefined` against `api.moltrust.ch` -- the same missing-top-level-
  `url` card shape that already broke ADK's and CrewAI's rigid
  pydantic/protobuf models earlier this session, here crashing
  Mastra's own URL construction instead of erroring cleanly.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from app import llm
from app.a2a_wire import _normalize, explain_error

MAX_TURNS = 4
CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}

# Layer-specific explanation for this module's own documented finding #4
# (see module docstring): run live at full scale, two raw unhandled JS
# exceptions turned up from the mastra_bridge.mjs subprocess itself,
# rather than a clean, attributed A2A-level error -- confirmed live
# against agent.co-legal.be/getamber.dev/postalform.com (a generic
# undefined-property crash) and api.moltrust.ch (a card missing its
# top-level `url`, crashing Mastra's own URL construction). Checked
# before falling back to the generic a2a_wire.explain_error() buckets,
# same precedent as adkgo_prober.py's own _explain_bridge_error().
_LAYER_ERROR_EXPLANATIONS = [
    (
        "Failed to parse URL from undefined",
        "Mastra's A2AAgent client crashed constructing a request URL, rather than reporting a "
        "clean error -- this happens when a target's card is missing a field (like a top-level "
        "\"url\") that this client assumes is always present. A real limitation of Mastra's own "
        "client, not necessarily a bug in the target agent's card.",
    ),
    (
        "Cannot read properties of undefined (reading 'message')",
        "Mastra's A2AAgent client crashed with a raw, unattributed JavaScript TypeError instead of "
        "a clean A2A-level error -- confirmed against several real agents whose response shape this "
        "client doesn't handle gracefully. A real limitation of Mastra's own client's error "
        "handling, not necessarily a bug in the target agent.",
    ),
    (
        ".well-known",
        # Only appears in the raw text when mastra_bridge.mjs's own
        # _formatError() appended a failing URL from MastraA2AError's
        # `.data.url` field (see the bridge's own comment) -- confirmed
        # by reading @mastra/core's compiled source directly
        # (dist/a2a/index.js line ~450: getAgentCard() only ever
        # constructs the current spec's well-known path, no fallback).
        "Mastra's A2AAgent client only tries the current A2A spec's well-known path "
        "(/.well-known/agent-card.json) -- it has no fallback to the older, legacy path "
        "(/.well-known/agent.json) that some real agents (like p0stman.com) still use exclusively. "
        "This project's own broader discovery already found this agent's card successfully (that's "
        "how its skills were enumerated for this report), which proves the card exists -- this is a "
        "limitation of Mastra's own client, not a genuinely undiscoverable agent.",
    ),
]


def _explain_layer_error(raw: str) -> str:
    for substring, explanation in _LAYER_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    return explain_error(raw)

BRIDGE_SCRIPT = Path(__file__).parent / "mastra_bridge.mjs"


class MastraBridge:
    """Manages one persistent `node mastra_bridge.mjs` subprocess."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            "node", str(BRIDGE_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def call(self, command: dict) -> dict:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write((json.dumps(command) + "\n").encode())
        await self._proc.stdin.drain()
        line = await self._proc.stdout.readline()
        if not line:
            raise RuntimeError("mastra_bridge.mjs process closed stdout unexpectedly (crashed?)")
        return json.loads(line.decode())

    async def close(self) -> None:
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.close()
            await self._proc.wait()


async def resolve_bridge(base_url: str) -> tuple[MastraBridge | None, dict | None, str | None]:
    """Returns (bridge, card_info, error). Starts the persistent Node
    process and initializes the A2AAgent via an "init" command."""
    bridge = MastraBridge()
    await bridge.start()
    result = await bridge.call({"cmd": "init", "base_url": base_url})
    if not result.get("ok"):
        await bridge.close()
        return None, None, _explain_layer_error(result.get("error", "unknown init error"))
    return bridge, {"name": result.get("name"), "url": result.get("url")}, None


async def probe_skill_mastra(bridge: MastraBridge, skill: dict) -> dict:
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
    run_id = str(uuid.uuid4())

    next_text = probe_text
    sent_desc = probe_text
    cmd = "generate"

    while turn < MAX_TURNS:
        try:
            result = await bridge.call({"cmd": cmd, "runId": run_id, "text": next_text})
        except Exception as e:
            error = _explain_layer_error(f"turn {turn + 1}: {type(e).__name__}: {e}")
            break
        turn += 1

        if not result.get("ok"):
            error = _explain_layer_error(f"turn {turn}: {result.get('error', 'unknown bridge error')}")
            break

        raw = result.get("task") or result.get("message")
        if raw is None:
            error = f"turn {turn}: mastra bridge returned neither task nor message"
            break

        normalized = await _normalize("1.0", raw, None, None)
        state = normalized["state"]
        text = normalized["agent_text"]
        states.append(state)
        messages.append(text)
        turns_log.append({"sent": sent_desc, "state": state, "received": text})
        artifacts.extend(normalized["artifacts"])
        final_state = state
        final_message = text

        if state in CONTINUABLE_STATES:
            if turn >= MAX_TURNS:
                error = f"did not reach a terminal state within {MAX_TURNS} turns"
                break
            followup_text = await llm.generate_followup(text, skill)
            next_text = followup_text
            sent_desc = followup_text
            cmd = "resume"
            continue
        elif state in POLLABLE_STATES:
            error = (
                "task is still working/submitted -- polling was not exercised "
                "by this check (no reachable agent in the current set naturally "
                "exercises it); a real caller would need to build that separately"
            )
            break
        else:
            break

    return {
        "turn": turn, "states": states, "messages": messages, "turns_log": turns_log,
        "artifacts": artifacts, "final_state": final_state, "final_message": final_message,
        "error": error,
    }
