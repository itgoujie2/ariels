"""Drives one skill through PraisonAI's own A2A client
(`praisonai.endpoints.providers.a2a.A2AProvider`) to a stopping point,
at parity with the golden probers' and the other native-client layers'
per-skill test depth.

Confirmed via direct package inspection: `praisonai` ships two
completely different things both named "A2A". `praisonai/capabilities/
a2a.py` (`a2a_send`/`aa2a_send`) is **not a real A2A protocol client at
all** -- reading its source directly shows it instantiates a brand-new
*local* `praisonaiagents.Agent` and calls `.chat()` on it in-process,
zero HTTP, zero wire protocol -- an internal multi-agent handoff
mechanism mislabeled "A2A", not real cross-framework interop. Ruled out.
`praisonai/endpoints/providers/a2a.py` (`A2AProvider`) is the genuine
one: a real JSON-RPC-over-HTTP client, part of PraisonAI's internal
multi-protocol gateway abstraction (alongside `provider_type`s for
"recipe", "agents-api", "mcp", etc.) rather than a dedicated
general-purpose A2A client class -- rougher and more internal-tool-
flavored than every other client tested this session, which is itself
part of the finding. Before settling on this, confirmed via direct
package inspection that Griptape, ControlFlow, swarms, and Mirascope
all ship zero A2A code of any kind.

Key findings, all confirmed live before/while building -- this is the
most restrictive client tested this session by a wide margin:

- **Decisive real finding: `A2AProvider`'s own outgoing request is not
  spec-conformant.** `invoke()`/`invoke_stream()` build message parts as
  `{"type": "text", "text": ...}` -- the A2A spec's discriminator field
  is `"kind"`, not `"type"`. Confirmed live against `2s.io` (a real,
  standards-conformant agent): `-32602 Invalid params: "message" must
  contain at least one text part` -- the target's own parser doesn't
  recognize the part at all, so the message is rejected outright before
  any real logic runs. This is a bug in the client's own *outgoing*
  request construction, not a response-parsing gap like every other
  finding this session -- unlike those, this prober does **not** work
  around it, since doing so would mean testing a hypothetical fixed
  client rather than the real, shipped one. Real capability is measured
  exactly as a caller using this class unmodified would experience it.
- **Real finding: no card-based endpoint resolution at all -- the RPC
  path is hardcoded to `{base_url}/a2a`,** regardless of what a target's
  own card declares as its actual endpoint (which can be any arbitrary
  path or even a different domain, as this project has repeatedly
  found). This prober therefore points `base_url` at each target's own
  domain root, the same naive-first-attempt choice a real developer
  with no special knowledge would make.
- **Real finding: card discovery only tries the *legacy* well-known path
  (`/.well-known/agent.json`), never the current spec path
  (`/.well-known/agent-card.json`) at all.** This is the *opposite* gap
  direction from every other client tested this session (all default to
  the current path and needed a legacy fallback built for them) --
  `A2AProvider` would fail to discover any agent that only serves the
  current path, a real and growing share of this project's dataset.
- **Real finding: no `task_id`/`context_id` field appears anywhere in
  the outgoing message** -- zero continuation support, the same class
  of gap already found in Strands' and Agno's clients. This prober
  therefore sends exactly **one turn per skill**.
- Synchronous-only (`invoke()`'s HTTP call and `invoke_stream()`'s raw
  `urllib.request` loop both block) -- wrapped in `asyncio.to_thread()`
  here to fit this project's async pipeline, the same accommodation a
  real caller integrating this into an async application would need.
"""

from __future__ import annotations

import asyncio

from praisonai.endpoints.providers.a2a import A2AProvider

from app import llm
from app.a2a_wire import _normalize, explain_error

CONTINUABLE_STATES = {"input-required"}
POLLABLE_STATES = {"working", "submitted"}

# Layer-specific explanation for this module's own documented, decisive
# finding (see module docstring): A2AProvider's own outgoing request
# builds message parts as {"type": "text", ...} instead of the spec's
# {"kind": "text", ...} discriminator -- a real, standards-conformant
# agent (confirmed live: 2s.io) rejects it outright with a JSON-RPC
# -32602 "Invalid params" naming the missing text part. A2AProvider's
# own invoke() only ever preserves the error *message* text, not the
# numeric JSON-RPC code (confirmed by reading its source), so this
# matches on the message's own distinctive wording rather than "-32602".
# Checked before falling back to the generic a2a_wire.explain_error()
# buckets, same precedent as adkgo_prober.py's own _explain_bridge_error().
_LAYER_ERROR_EXPLANATIONS = [
    (
        "must contain at least one text part",
        "PraisonAI's A2AProvider builds its outgoing message parts using \"type\" as the "
        "discriminator field instead of the A2A spec's \"kind\" -- a standards-conformant agent "
        "doesn't recognize the part at all and rejects the request outright before any real logic "
        "runs. A real limitation of A2AProvider's own outgoing request construction, not an issue "
        "with the target agent.",
    ),
]


# A2AProvider's hardcoded `/a2a` RPC path (see module docstring finding
# #2) means a target whose real endpoint lives elsewhere often just
# gets a plain website 404 back, not a JSON-RPC error. PraisonAI's own
# `_make_request()` (praisonai/endpoints/providers/base.py) doesn't
# check for this: on a non-2xx response it tries `json.loads(body)`,
# and on a JSONDecodeError falls back to `{"message": body}` --
# `body` being *the entire raw response text*. Confirmed live against
# p0stman.com (a Next.js site): a plain 404 page's full HTML,
# including every <script>/<link> tag, ends up as the error message
# verbatim. Detected structurally (does the raw text look like an HTML
# document?) rather than matched by any specific substring, since the
# actual page content is different for every target this hits, and
# truncated separately from the normal "(raw error: ...)" suffix --
# appending the full multi-KB page would make the report less
# readable, not more.
_HTML_BODY_MARKERS = ("<!DOCTYPE", "<html")
_HTML_BODY_PREVIEW_CHARS = 200


def _explain_layer_error(raw: str) -> str:
    for substring, explanation in _LAYER_ERROR_EXPLANATIONS:
        if substring in raw:
            return f"{explanation} (raw error: {raw})"
    if any(marker.lower() in raw[:500].lower() for marker in _HTML_BODY_MARKERS):
        preview = raw[:_HTML_BODY_PREVIEW_CHARS]
        if len(raw) > _HTML_BODY_PREVIEW_CHARS:
            preview += f"... ({len(raw)} chars total)"
        return (
            "PraisonAI's A2AProvider hardcodes its RPC endpoint to \"{base_url}/a2a\" regardless of "
            "what a target agent actually declares -- when that guessed path doesn't correspond to a "
            "real A2A endpoint, a normal website returns its own 404 page (a full HTML document) "
            "instead of a JSON-RPC error. A2AProvider's own error handling doesn't check for this: it "
            "treats the entire HTML response body as if it were the error message text. A real "
            "limitation of A2AProvider's own client (the hardcoded endpoint guess, plus its non-JSON-"
            f"body handling), not a bug in the target agent. (raw HTML preview: {preview})"
        )
    return explain_error(raw)


async def probe_skill_praisonai(provider: A2AProvider, skill: dict) -> dict:
    """Returns a transcript dict shaped exactly like the golden probers'
    and the other native-client layers', so
    `app.assertions.check_transcript()` applies unchanged.

    Always exactly one turn -- see module docstring: `A2AProvider` has
    no way to pass a `task_id`/`context_id` back for continuation at
    all, so `input-required`/`auth-required` is reported as a legitimate
    blocked terminal state on turn 1, not a failed follow-up."""
    probe_text = skill["examples"][0] if skill.get("examples") else await llm.generate_probe(skill)

    try:
        result = await asyncio.to_thread(
            provider.invoke, name="__probe__", input_data={"message": probe_text}
        )
    except Exception as e:
        return {
            "turn": 0, "states": [], "messages": [], "turns_log": [],
            "artifacts": [], "final_state": None, "final_message": None,
            "error": _explain_layer_error(f"turn 1: {type(e).__name__}: {e}"),
        }

    if not result.ok:
        return {
            "turn": 1, "states": [], "messages": [], "turns_log": [],
            "artifacts": [], "final_state": None, "final_message": None,
            "error": _explain_layer_error(str(result.error)),
        }

    try:
        normalized = await _normalize("1.0", result.data, None, None)
    except Exception as e:
        # Real finding (see module docstring): when the hardcoded
        # `{base_url}/a2a` path doesn't correspond to a target's real
        # endpoint, A2AProvider can come back `ok=True` with an empty
        # `data` dict instead of a clear error -- confirmed live against
        # insideout.luthersystems.com. Surfaced here as an attributed
        # error rather than crashing the run.
        return {
            "turn": 1, "states": [], "messages": [], "turns_log": [],
            "artifacts": [], "final_state": None, "final_message": None,
            "error": _explain_layer_error(f"{type(e).__name__}: {e}"),
        }
    state = normalized["state"]
    text = normalized["agent_text"]

    turn_error = None
    if state in POLLABLE_STATES:
        turn_error = (
            "task is still working/submitted -- polling a task by id was not "
            "exercised by this check (A2AProvider has no tasks/get equivalent "
            "at all); a real caller would need to build that separately"
        )

    return {
        "turn": 1,
        "states": [state],
        "messages": [text],
        "turns_log": [{"sent": probe_text, "state": state, "received": text}],
        "artifacts": normalized["artifacts"],
        "final_state": state,
        "final_message": text,
        "error": turn_error,
    }
