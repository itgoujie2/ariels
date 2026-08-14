"""Drives one skill through Google ADK for Go's own first-party A2A
client (`google.golang.org/adk/agent/remoteagent/v2`'s `NewA2A`) to a
stopping point, at parity with the golden probers' and the other
native-client layers' per-skill test depth -- via a cross-language
bridge, since ADK-Go is Go, not Python (the second cross-language layer
in this matrix, after Mastra/JS).

**Architecture note**: `bridge/adkgo_bridge` is a persistent Go
subprocess (compiled ahead of time, spawned once per skill, kept alive
for the whole multi-turn probe) that holds one real ADK `runner.Runner`
+ `session.InMemoryService` + remote `NewA2A` agent, and speaks
newline-delimited JSON over stdin/stdout -- the exact same pattern as
the Mastra layer's `mastra_bridge.mjs`, for the same reason: ADK's own
session/run state lives in that process's memory and can't be
reconstructed from a fresh process. This prober drives it turn by turn
like every other layer, then feeds the raw `a2a:response` JSON straight
through `app.a2a_wire._normalize()` -- ADK-Go's own event metadata
already carries the real underlying A2A wire response (confirmed live:
`event.CustomMetadata["a2a:response"]` is a plain map matching the wire
shape), the same reuse principle every other layer follows.

Key findings, all confirmed live before/while building:

- **Real finding: `NewA2A`'s remote agent requires a card's
  `supportedInterfaces` field to be present at all** -- confirmed live
  against `2s.io` (whose card is a flat, `supportedInterfaces`-less
  v0.3-style card, the shape a large share of this project's dataset
  uses): `sender creation failed: agent card has no supported
  interfaces`, a clean but immediate dead end. Every other client
  tested this session at minimum falls back to a flat `url` field for
  v0.3-style discovery; this is the first to hard-require the more
  specific field with zero fallback.
- **Real finding, and the most severe continuation gap of the whole
  session**: for a real agent replying to `input-required` with plain
  text (the overwhelming majority of real agents, including every one
  used to probe this session's other clients) there is **no
  continuation mechanism through this client at all**. Reading the
  source (`server/adka2a/v2/events.go`'s `taskToEvent`,
  `agent/remoteagent/v2/utils.go`'s `getUserFunctionCallAt`) shows
  continuation is only wired for the A2A "long-running tool" extension
  (a structured `DataPart`-based function-call convention) -- a plain
  text `input-required` reply never gets a synthetic `FunctionCall`
  constructed for it at all, unlike the *Python* ADK port's
  `RemoteA2aAgent` (which does synthesize one generically, see
  `native-client-checks/`'s own docstring). Confirmed live against
  `insideout.luthersystems.com`: calling `Run()` again on the exact same
  session with a follow-up message does not continue the task --
  turn 2's `task_id` *and* `context_id` both differ from turn 1's, the
  same "silent fresh task" symptom already found in CrewAI's, Agno's,
  and Mastra's clients, but here there is no API surface that even
  *pretends* to support resuming a plain-text conversation at all. This
  prober still attempts the natural, only-available thing a real
  caller would try (send the follow-up on the same session) and
  documents the result, exactly like the Mastra layer's finding.
- **Real finding: card parsing rejects several real, standard security
  scheme shapes outright.** Confirmed live at full batch scale against
  four different ground-truth `fully_working` agents (`a2a.decision-
  anchor.com`, `agent.co-legal.be`, `api.moltrust.ch`, `getamber.dev`):
  `card parsing failed: unknown security scheme type for <name>: [...]`
  -- a discriminator-matching failure on standard `bearer`/`apiKey`
  security scheme objects that every other client tested this session
  parses without incident.
- **Real finding: a second, distinct transport-negotiation failure** --
  confirmed against `hello.a2aregistry.org` (the official reference
  Hello World A2A sample): `sender creation failed: no compatible
  transports found: available transports - [_]`, a clean rejection of
  a card this project's own tolerant parser and every other client
  handles fine.
- **Real finding: a third, distinct parsing failure on an alternate
  streaming-event shape** -- confirmed against
  `paki-api.elfresonero.workers.dev` (already known from earlier this
  session to reply in pure v0.3-flat shape regardless of request
  dialect): `unknown event type: [kind taskId contextId status final]`,
  a shape this client's stricter event-type discriminator doesn't
  recognize at all.
- Taken together with the `supportedInterfaces` requirement and the
  missing plain-text continuation path, this makes ADK-Go's client the
  **most restrictive client tested this whole session**: run live at
  full scale (153 agents), only **2/153 reachable**, and 9 of this
  project's 10 ground-truth `fully_working` agents are unreachable
  through it -- tied with PraisonAI's for the highest divergence found.
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

BRIDGE_BINARY = Path(__file__).parent / "bridge" / "adkgo_bridge"

# Plain-language lead-in for the raw adkgo_bridge error strings this
# module's own docstring already documents as known, real findings --
# confirmed live, a real user reading a report found the raw Go error
# text ("sender creation failed: agent card has no supported
# interfaces") unclear on its own, with no indication of whether this
# was a bug in Ariel or in the target agent's card. Matched by substring
# against the bridge's own fixed error-message shapes (not something
# this client varies per target), same "wrap the raw error, don't
# discard it" approach already used for the AI-Judgment layer's own
# dialect-mismatch errors.
_BRIDGE_ERROR_EXPLANATIONS = [
    (
        "no supported interfaces",
        "ADK-Go's client requires a card to explicitly declare a `supportedInterfaces` field -- this "
        "target's card uses the older, more common flat `url` style instead, which most other clients "
        "accept but ADK-Go doesn't fall back to. This is a real limitation of ADK-Go's own client, not "
        "an issue with the target agent's card.",
    ),
    (
        "no compatible transports found",
        "ADK-Go's client couldn't negotiate a transport it recognizes with this agent, even though "
        "this project's own parser and every other client tested handle its card fine. A real "
        "limitation of ADK-Go's own transport negotiation, not an issue with the target agent.",
    ),
    (
        "unknown security scheme type",
        "ADK-Go's client rejected this card's security-scheme declaration outright -- a standard "
        "bearer/apiKey shape every other client tested parses without incident. A real limitation of "
        "ADK-Go's own card parsing, not an issue with the target agent's card.",
    ),
    (
        "unknown event type",
        "ADK-Go's client received a response shape it doesn't recognize, even though this project's "
        "own parser and other clients handle it fine. A real limitation of ADK-Go's own event parsing, "
        "not an issue with the target agent's response.",
    ),
    (
        "failed to fetch an agent card",
        "ADK-Go's client only tries the current A2A spec's well-known path (/.well-known/agent-card.json) "
        "-- it has no fallback to the older, legacy path (/.well-known/agent.json) that some real agents "
        "(like p0stman.com) still use exclusively. This project's own broader discovery already found "
        "this agent's card successfully (that's how its skills were enumerated for this report), which "
        "proves the card exists -- this is a limitation of ADK-Go's own client, not a genuinely "
        "undiscoverable agent.",
    ),
]


def _explain_bridge_error(raw: str) -> str:
    for substring, explanation in _BRIDGE_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    # Part of a cross-cutting fix: an error that isn't one of this
    # layer's own 4 documented bridge-specific shapes may still be a
    # generic, cross-layer-recognizable one (dialect mismatch, auth,
    # payment, internal error) -- fall back to the shared classifier
    # instead of returning raw text unexplained.
    return explain_error(raw)


class AdkGoBridge:
    """Manages one persistent `adkgo_bridge` Go subprocess."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            str(BRIDGE_BINARY),
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
            raise RuntimeError("adkgo_bridge process closed stdout unexpectedly (crashed?)")
        return json.loads(line.decode())

    async def close(self) -> None:
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.close()
            await self._proc.wait()


async def resolve_bridge(base_url: str) -> tuple[AdkGoBridge | None, str | None]:
    """Returns (bridge, error). Starts the persistent Go process and
    initializes the remote A2A agent via an "init" command."""
    bridge = AdkGoBridge()
    await bridge.start()
    result = await bridge.call({"cmd": "init", "base_url": base_url})
    if not result.get("ok"):
        await bridge.close()
        return None, _explain_bridge_error(result.get("error", "unknown init error"))
    return bridge, None


async def probe_skill_adkgo(bridge: AdkGoBridge, skill: dict) -> dict:
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
    session_id = str(uuid.uuid4())

    next_text = probe_text
    sent_desc = probe_text
    cmd = "generate"

    while turn < MAX_TURNS:
        try:
            result = await bridge.call({"cmd": cmd, "sessionId": session_id, "text": next_text})
        except Exception as e:
            error = explain_error(f"turn {turn + 1}: {type(e).__name__}: {e}")
            break
        turn += 1

        if not result.get("ok"):
            error = f"turn {turn}: {_explain_bridge_error(result.get('error', 'unknown bridge error'))}"
            break

        raw = result.get("response")
        if raw is None:
            error = f"turn {turn}: adk-go bridge returned no a2a response"
            break

        normalized = await _normalize("1.0", raw, result.get("taskId"), result.get("contextId"))
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
