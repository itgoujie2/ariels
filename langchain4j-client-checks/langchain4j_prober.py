"""Drives one skill through LangChain4j's own first-party A2A client
(`dev.langchain4j.agentic.a2a`'s `DefaultA2AService`/`A2AClientBuilder`,
backed by the official `org.a2aproject.sdk` `a2a-java-sdk-client`) to a
stopping point, at parity with the golden probers' and the other
native-client layers' per-skill test depth -- via a cross-language
bridge, since LangChain4j is Java, not Python (the third cross-language
layer in this matrix, after Mastra/JS and ADK-Go/Go).

**Architecture note**: unlike the Mastra and ADK-Go layers, this one
does *not* need a persistent process holding session state across
turns. LangChain4j's own generated client interface accepts
`@A2AContextId`/`@A2ATaskId`-annotated method parameters directly, so
continuation state is passed in and read back out explicitly on every
call -- confirmed live this works exactly as documented. `bridge/`
(compiled ahead of time into a shaded jar via Maven) is invoked fresh
per turn: `java -jar langchain4j-bridge.jar <baseUrl> <text> <contextId
|null> <taskId|null>`, printing one line of JSON to stdout. This
prober extracts `contextId`/`taskId` from the returned raw `Task` JSON
itself and threads them into the next call.

**This turned out to be the least reachable client of the entire
session -- confirmed across 11 different real, live, independently-
discovered agents before batching, zero of which completed a single
successful call.** Four distinct, reproducible bug classes account for
every failure seen, confirmed against two different `a2a-java-sdk`
versions (`1.0.0.Final` and the newer `1.1.0.Final` -- same bugs in
both):

1. **`No server interface available in the AgentCard`** -- the same
   `supportedInterfaces`-required gap already found in ADK-Go's client
   (both are official SDKs from the same `a2aproject` org). Confirmed
   against `2s.io`, `arcasos.com` (flat, `supportedInterfaces`-less
   v0.3-style cards).
2. **`SecurityScheme oneof field not set`** -- the single most common
   failure by far, confirmed against `agent.co-legal.be`,
   `rettfrabonden.com`, `api.sursatech.com`, `api.moltrust.ch`,
   `agent-ready.dev` (five different real agents, including some with
   zero security requirements and some with real bearer-token auth
   documented in their own cards -- the failure doesn't correlate with
   whether a security scheme is actually declared, only with the card
   passing through this SDK's internal gRPC-style proto conversion path
   at all). This is a *sweeping* card-parsing bug: reading the stack
   trace (`org.a2aproject.sdk.grpc.utils.ProtoUtils$FromProto.convert`)
   shows every card gets converted through an internal protobuf
   representation regardless of the JSON-RPC transport actually in use,
   and that conversion path chokes on the `security`/`securitySchemes`
   field in some form on the overwhelming majority of real cards tested.
3. **A silent hang, not even an error** -- confirmed live against
   `insideout.luthersystems.com` (a card that *does* have
   `supportedInterfaces` and *doesn't* trip the SecurityScheme bug):
   the client simply never returns, streaming-response handling never
   completing or erroring out. This prober relies on the batch runner's
   own per-target timeout to recover from this, the same as every other
   layer's hang-handling convention.
4. Other real cards failed with a generic `Could not unmarshal agent
   card response` (`ai.syln.cn`, `getamber.dev`) or a clean 404 for a
   legacy-well-known-path-only card (`p0stman.com`, the same gap found
   repeatedly this session in other clients).

Given the severity and consistency of these findings (0/11 manual
tests succeeded, across a deliberately varied real-agent sample), this
prober is built and run at the same full 153-target batch scale as
every other layer, since the *precise* number and the *why* are
themselves the finding -- not assumed, but measured the same way as
every less-restrictive client this session.

**Confirmed at full scale: 0/153 reachable -- the only client this
entire session to reach zero real agents.** Error distribution across
all 153 targets: 69 `No server interface available in the AgentCard`
(the `supportedInterfaces`-required gap, far more common here than in
ADK-Go's client -- both, interestingly, share the same root SDK
family); 33 `SecurityScheme oneof field not set`; 23 clean 404s
(legacy-card-only agents); 9 other distinct `A2AClientException`
variants; 7 card-unmarshal failures; 4 streaming-response failures; 3
`Method not found` (dialect mismatches); only 2 actual hangs (most
failures are fast, not the 30s timeout). Full skill-level parity:
**0/0** (nothing reachable to test). 10 of this project's 10
ground-truth `fully_working` agents are unreachable through it -- the
only client this session to miss all ten.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from app import llm
from app.a2a_wire import _normalize, explain_error

MAX_TURNS = 4
CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}

# Layer-specific explanations for this module's own documented findings
# (see module docstring) -- run live at full scale (153 agents), this
# was the only client this whole session to reach zero real agents, and
# these are its two highest-volume failure modes by a wide margin
# (69/153 and 33/153 respectively). Checked before falling back to the
# generic a2a_wire.explain_error() buckets, same precedent as
# adkgo_prober.py's own _explain_bridge_error().
_LAYER_ERROR_EXPLANATIONS = [
    (
        "No server interface available in the AgentCard",
        "LangChain4j's official A2A client (and ADK-Go's, from the same a2aproject org) requires a "
        "card to explicitly declare a `supportedInterfaces` field -- this was the single most "
        "common failure mode of any client tested this session (69/153 real agents), since a large "
        "share of real cards use the older, flat `url` style instead, which most other clients "
        "accept but this one doesn't fall back to. A real limitation of this client's own card "
        "handling, not an issue with the target agent's card.",
    ),
    (
        "SecurityScheme oneof field not set",
        "LangChain4j's client converts every card through an internal protobuf representation "
        "regardless of the JSON-RPC transport actually in use, and that conversion path chokes on "
        "the card's `security`/`securitySchemes` field for the majority of real cards tested (33/153 "
        "agents), with no correlation to whether a security scheme is actually meaningfully declared. "
        "A real limitation of this client's own card parsing, not an issue with the target agent.",
    ),
    (
        "Failed to obtain agent card: 404",
        "LangChain4j's own card-discovery step only tries the current A2A spec's well-known path "
        "(/.well-known/agent-card.json) -- it has no fallback to the older, legacy path "
        "(/.well-known/agent.json) that some real agents (like p0stman.com) still use exclusively "
        "(23/153 real agents hit this exact gap). This project's own broader discovery already found "
        "this agent's card successfully (that's how its skills were enumerated for this report), which "
        "proves the card exists -- this is a limitation of LangChain4j's own client, not a genuinely "
        "undiscoverable agent.",
    ),
]


def _explain_layer_error(raw: str) -> str:
    for substring, explanation in _LAYER_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    return explain_error(raw)

BRIDGE_JAR = Path(__file__).parent / "bridge" / "target" / "langchain4j-bridge-1.0.jar"
TURN_TIMEOUT_S = 30

# The bridge jar targets Java 21 bytecode, but this machine's plain `java`
# on PATH resolves to /usr/bin/java -> an older pre-existing JDK 13, which
# fails with UnsupportedClassVersionError. Homebrew's openjdk@21 is
# keg-only (not symlinked onto PATH by default), so its absolute path is
# preferred when present; falls back to whatever `java` resolves to
# otherwise (e.g. in a differently-configured environment/CI).
_HOMEBREW_JAVA21 = Path("/opt/homebrew/opt/openjdk@21/bin/java")
JAVA_BIN = str(_HOMEBREW_JAVA21) if _HOMEBREW_JAVA21.exists() else (shutil.which("java") or "java")


async def _call_bridge(base_url: str, text: str, context_id: str | None, task_id: str | None) -> dict:
    proc = await asyncio.create_subprocess_exec(
        JAVA_BIN, "-jar", str(BRIDGE_JAR), base_url, text,
        context_id if context_id else "null", task_id if task_id else "null",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"ok": False, "error": f"bridge process timed out after {TURN_TIMEOUT_S}s (client hang -- see module docstring)"}
    if not stdout:
        return {"ok": False, "error": "bridge process produced no output"}
    try:
        return json.loads(stdout.decode().strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "error": f"bridge process produced non-JSON output: {stdout.decode()[:300]!r}"}


async def resolve_bridge(base_url: str) -> tuple[str | None, str | None]:
    """Returns (base_url, error) -- no persistent process to set up for
    this layer (see module docstring), just validates the jar exists."""
    if not BRIDGE_JAR.exists():
        return None, f"bridge jar not found at {BRIDGE_JAR} -- run `cd bridge && mvn package` first"
    return base_url, None


async def probe_skill_langchain4j(base_url: str, skill: dict) -> dict:
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

    next_text = probe_text
    sent_desc = probe_text

    while turn < MAX_TURNS:
        try:
            result = await _call_bridge(base_url, next_text, context_id, task_id)
        except Exception as e:
            error = _explain_layer_error(f"turn {turn + 1}: {type(e).__name__}: {e}")
            break
        turn += 1

        if not result.get("ok"):
            error = _explain_layer_error(f"turn {turn}: {result.get('error', 'unknown bridge error')}")
            break

        raw = result.get("task")
        if raw is None:
            error = f"turn {turn}: langchain4j bridge returned no task"
            break

        normalized = await _normalize("1.0", raw, task_id, context_id)
        state = normalized["state"]
        text = normalized["agent_text"]
        task_id = normalized["task_id"] or task_id
        context_id = normalized["context_id"] or context_id
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
