"""Hand-rolled, dialect-tolerant A2A wire layer.

Why not just use `a2a.client.A2AClient`? Because it's pinned to one SDK
version's idea of the wire format, and that turned out to matter: a live test
against the official, unmodified `helloworld` sample (built on the *current*
a2a-sdk 1.x) proved the wire protocol itself changed between spec/SDK major
versions, not just the Python API:

  v0.3 (our fixture, langchain-samples demo, a2a-sdk 0.3.x):
    - JSON-RPC method: "message/send"
    - role: "user" / "agent"
    - part: {"kind": "text", "text": "..."}
    - task state: "completed", "input-required", "auth-required" (kebab-case)
    - response envelope: {"result": {<task fields directly>}}

  v1.0 (current a2a-sdk 1.x, protobuf-based):
    - requires header "A2A-Version: 1.0" (omitting it defaults to 0.3 and
      the request is then rejected as version-incompatible)
    - JSON-RPC method: "SendMessage" (PascalCase, no slash)
    - role: "ROLE_USER" / "ROLE_AGENT"
    - part: {"text": "..."} (no "kind" discriminator)
    - task state: "TASK_STATE_COMPLETED", "TASK_STATE_INPUT_REQUIRED", ...
      (protobuf enum names — note "CANCELLED" double-L vs v0.3's "canceled")
    - response envelope: {"result": {"task": {<task fields>}}} — one level
      deeper than v0.3

`dialect` only controls how *requests* are built (_build_request) — which
method name and header to send. It does NOT predict response shape: two
real agents have now been confirmed replying in the *other* dialect's shape
regardless of which one was used to call them (agent.co-legal.be always
replies v1.0-style; paki-api.elfresonero.workers.dev always replies
v0.3-flat-style). So `_normalize` is shape-driven — it inspects the response
itself (`_task_from_result`/`_message_from_result`) rather than trusting
the dialect parameter to predict what it'll find.
"""

from __future__ import annotations

import json
import os
from uuid import uuid4

import httpx

from app import llm

KNOWN_DIALECTS = ["0.3", "1.0"]
PING_TEXT = "ping"  # minimal, harmless probe, not tied to any declared skill


class UnrecognizedResponseShapeError(RuntimeError):
    """Raised when a response is neither a recognizable Task nor a Message.

    Deliberately loud rather than silently falling through to a "bare
    message" guess: a response we can't confidently classify getting
    reported as e.g. an empty-but-'completed' task would look like a content
    gap in the target agent, when it's actually a coverage gap in this
    prober. Callers should catch this and record it as an explicit error,
    not let it masquerade as a normal result.
    """

    def __init__(self, dialect: str, result: dict):
        keys = sorted(result.keys()) if isinstance(result, dict) else type(result).__name__
        super().__init__(f"unrecognized {dialect} response shape (top-level keys: {keys})")
        self.dialect = dialect
        self.result = result

V1_STATE_MAP = {
    "TASK_STATE_UNSPECIFIED": "unknown",
    "TASK_STATE_SUBMITTED": "submitted",
    "TASK_STATE_WORKING": "working",
    "TASK_STATE_COMPLETED": "completed",
    "TASK_STATE_FAILED": "failed",
    "TASK_STATE_CANCELLED": "canceled",
    "TASK_STATE_INPUT_REQUIRED": "input-required",
    "TASK_STATE_REJECTED": "rejected",
    "TASK_STATE_AUTH_REQUIRED": "auth-required",
}


CARD_PATHS = [
    "/.well-known/agent-card.json",  # current spec path
    "/.well-known/agent.json",  # legacy path, predates the agent-card.json rename
]


class NoAgentCardFoundError(RuntimeError):
    """Raised when no known well-known path returns something that actually
    looks like an agent card."""


def _looks_like_agent_card(data) -> bool:
    """A minimal sanity check: real agent cards always have a `name`.

    Confirmed live (insideout.luthersystems.com): a misconfigured discovery
    endpoint can return HTTP 200 with a *JSON-RPC error envelope* instead of
    404ing — `{"jsonrpc": "2.0", "error": {...}}` — rather than a card.
    Without this check, fetch_card would silently accept that as "the card"
    and produce a confusing downstream failure (empty skills, null name,
    then an unrelated-looking RPC error) instead of a clear, attributed
    "couldn't discover this agent" signal at the point where it's obvious.
    """
    return isinstance(data, dict) and "name" in data and "jsonrpc" not in data


async def fetch_card(httpx_client: httpx.AsyncClient, base_url: str) -> dict:
    """Tries every known well-known path, not just the current one.

    Found live: p0stman.com/Zee (a real, live production agent) only serves
    its card at the legacy `/.well-known/agent.json` path — the current
    `/.well-known/agent-card.json` 404s. This is discovery-layer version
    skew, the same phenomenon as the wire-protocol dialect split, just one
    layer earlier: before a single message is ever sent, a naive client
    hardcoding only the current path can't even find the agent."""
    base = base_url.rstrip("/")
    attempts = []
    for path in CARD_PATHS:
        try:
            resp = await httpx_client.get(f"{base}{path}")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            attempts.append(f"{path}: HTTP {e.response.status_code}")
            continue
        except ValueError as e:  # not valid JSON at all
            attempts.append(f"{path}: invalid JSON ({e})")
            continue
        if _looks_like_agent_card(data):
            return data
        attempts.append(f"{path}: HTTP 200 but not a valid agent card (keys: {sorted(data.keys()) if isinstance(data, dict) else type(data).__name__})")
    raise NoAgentCardFoundError(f"no known well-known path returned a valid agent card: {'; '.join(attempts)}")


def resolve_target(card: dict, base_url: str) -> tuple[str, str]:
    """Returns (rpc_url, declared_dialect).

    `declared_dialect` is purely informational now — run_prober.py always
    probes with the latest dialect (see LATEST_DIALECT there) regardless of
    what a card claims to speak, and reports "doesn't support latest" as a
    real finding rather than adapting to match. This function still matters
    for finding the right rpc_url, and for the declared/tested contrast in
    the report.

    `supportedInterfaces` is checked first, even when a flat `url` is also
    present: confirmed live (agent.co-legal.be) that a card can carry both —
    a legacy flat `url` plus a `supportedInterfaces` entry declaring the
    real (newer) protocolVersion — and the flat `url`'s implied "0.3.0"
    default is the wrong one. `supportedInterfaces` is the more specific,
    per-transport declaration, so it wins when present.
    """
    for iface in card.get("supportedInterfaces", []):
        binding = (iface.get("protocolBinding") or "").upper()
        if binding in ("JSONRPC", "JSON-RPC", "JSON_RPC"):
            return iface.get("url", card.get("url", base_url)), iface.get("protocolVersion", "1.0")[:3]

    if "url" in card:
        return card["url"], card.get("protocolVersion", "0.3.0")[:3]

    return base_url, "0.3"


def get_skills(card: dict) -> list[dict]:
    skills = [
        {
            "id": s.get("id", "unknown"),
            "name": s.get("name", s.get("id", "unknown")),
            "description": s.get("description", ""),
            "examples": list(s.get("examples") or []),
        }
        for s in card.get("skills") or []
    ]
    # Unset (the default) preserves existing behavior exactly -- this is
    # only ever set by product/'s orchestrator, per-invocation, to bound
    # worst-case cost for a public on-demand run (a real agent in this
    # project's own dataset declared 120 skills and crashed one client's
    # internal stack-depth limit). Never touches offline batch runs.
    max_skills = os.environ.get("ARIEL_MAX_SKILLS")
    if max_skills:
        skills = skills[: int(max_skills)]
    return skills


def _build_message(dialect: str, text: str, task_id: str | None, context_id: str | None) -> tuple[str, dict]:
    message_id = uuid4().hex
    if dialect == "1.0":
        message = {"role": "ROLE_USER", "parts": [{"text": text}], "messageId": message_id}
    else:
        message = {"role": "user", "parts": [{"kind": "text", "text": text}], "messageId": message_id}
    if task_id:
        message["taskId"] = task_id
    if context_id:
        message["contextId"] = context_id
    return message_id, message


def _build_request(dialect: str, text: str, task_id: str | None, context_id: str | None) -> dict:
    message_id, message = _build_message(dialect, text, task_id, context_id)
    method = "SendMessage" if dialect == "1.0" else "message/send"
    return {"jsonrpc": "2.0", "id": message_id, "method": method, "params": {"message": message}}


def _build_get_task_request(dialect: str, task_id: str) -> dict:
    """Per the reference SDK's own proto (a2a_v0_3.proto): v1.0's GetTaskRequest
    uses an AIP resource-name style field, `name: "tasks/{task_id}"`, while
    v0.3's TaskQueryParams uses a flat `id`. Same PascalCase-vs-slash method
    naming split as message/send vs SendMessage."""
    request_id = uuid4().hex
    if dialect == "1.0":
        params = {"name": f"tasks/{task_id}"}
        method = "GetTask"
    else:
        params = {"id": task_id}
        method = "tasks/get"
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def _build_stream_request(dialect: str, text: str) -> dict:
    """Confirmed live (our own fixture and postalform.com's rejection of it):
    v0.3 method is "message/stream"; v1.0 is "SendStreamingMessage" (per the
    current a2a-sdk's own JSON-RPC method catalog)."""
    message_id, message = _build_message(dialect, text, None, None)
    method = "SendStreamingMessage" if dialect == "1.0" else "message/stream"
    return {"jsonrpc": "2.0", "id": message_id, "method": method, "params": {"message": message}}


def _extract_texts(parts: list[dict]) -> list[str]:
    """Text parts extract directly. Data parts (`{"kind": "data", "data":
    {...}}`) are just as legitimate a real answer — confirmed live:
    agent-ready.dev's 'ask' skill returns a fully substantive, real answer
    (8 relevant Schema.org-typed results) entirely as a data part, no text
    part at all. Serialize it so content-presence checks don't treat a
    genuinely rich structured answer as empty."""
    out = []
    for p in parts:
        if p.get("text"):
            out.append(p["text"])
        elif p.get("data"):
            out.append(json.dumps(p["data"]))
    return out


def _looks_like_message(obj: dict) -> bool:
    """A Message is required by spec to carry `parts` (possibly empty) and a
    `role` — that's its defining signature, distinct from a Task."""
    return isinstance(obj, dict) and "parts" in obj


def _task_from_result(result: dict) -> dict | None:
    """Find the task object regardless of which envelope shape it's in:
    nested under `task` (v1.0-style) or the result itself if it's flat
    `kind: "task"` (v0.3-style).

    Deliberately does NOT branch on which dialect the *request* used.
    Confirmed live, in both directions, that response shape doesn't track
    request dialect: agent.co-legal.be replies in pure v1.0 shape even to a
    v0.3-style `message/send` call, while paki-api.elfresonero.workers.dev
    replies in pure v0.3-flat shape even to a v1.0 `SendMessage` call with
    the `A2A-Version` header set. Some real agents just have one canonical
    internal representation and serialize it the same way regardless of how
    they were asked — so the only reliable signal is the response's own
    shape, not the request that produced it."""
    if isinstance(result.get("task"), dict):
        task = result["task"]
        # Some agents (verified live: rettfrabonden.com) put "artifacts" as
        # a sibling of the nested "task" wrapper -- {"result": {"task":
        # {...}, "artifacts": [...]}} -- rather than inside it. Without
        # this, the artifacts were invisible to deterministic extraction
        # and silently fell through to the (slower, costlier) LLM
        # fallback for content that was right there one level up.
        if "artifacts" not in task and isinstance(result.get("artifacts"), list):
            task = {**task, "artifacts": result["artifacts"]}
        return task
    if result.get("kind") == "task":
        return result
    # Some agents (verified live: p0stman.com/Zee) omit the "kind"
    # discriminator entirely. `status` is a Task's defining signature
    # regardless — only Tasks have one; Messages carry `parts`/`role`
    # instead — so accept that as sufficient evidence when "kind" is absent.
    if "kind" not in result and "status" in result:
        return result
    return None


def _message_from_result(result: dict) -> dict | None:
    """Same shape-over-request-dialect principle as _task_from_result, for
    bare Message results (agent answered without ever opening a task)."""
    if isinstance(result.get("message"), dict):
        return result["message"]
    if result.get("kind") == "message" or _looks_like_message(result):
        return result
    return None


async def _content_or_llm_fallback(state: str, agent_text: str, artifacts: list[str], raw_obj: dict) -> str:
    """Deterministic field-checking (_extract_texts) is fast, free, and
    already covers every shape we've seen live. But real agents keep
    inventing new places to put an answer — confirmed three times now
    (artifacts-only, message-only, data-parts-only) — and hardcoding one
    more field path each time we find another is a losing game. If a
    'completed' response has genuinely nothing in any known field, ask the
    model to find it wherever it is, rather than reporting a false empty."""
    if state == "completed" and not agent_text.strip() and not any(a.strip() for a in artifacts):
        return await llm.extract_content(raw_obj)
    return agent_text


async def _normalize(dialect: str, result: dict, fallback_task_id, fallback_context_id) -> dict:
    task = _task_from_result(result)
    if task is not None:
        status = task.get("status", {})
        # Some agents (verified live: ai.syln.cn / "Kunlun") put the state
        # directly as a plain string -- {"status": "completed"} -- rather
        # than nesting it under a state field -- {"status": {"state": ...}}
        # -- as every other agent tested does. Normalize rather than crash
        # with a raw 'str' object has no attribute 'get' from deep inside
        # our own parsing code.
        if isinstance(status, str):
            status = {"state": status}
        state_raw = status.get("state", "unknown")
        # Passes through unchanged if already kebab-case (v0.3-style); maps
        # protobuf enum names (v1.0-style) back to the same casing. Either
        # can show up regardless of which dialect the request used.
        state = V1_STATE_MAP.get(state_raw, state_raw)
        # Some agents (verified live: getamber.dev / "Ambr") use underscores
        # instead of the spec's hyphens -- "input_required" instead of
        # "input-required". No legitimate state name uses an underscore, so
        # this is a safe, blanket normalization. Without it, this state was
        # invisible to both assertions.py's known-state checks AND, more
        # importantly, prober.py's CONTINUABLE_STATES check -- meaning the
        # prober never even attempted to resolve it via a follow-up, the
        # first real agent this whole session to naturally need one.
        state = state.replace("_", "-")
        # Same agent (ai.syln.cn) also puts the reply message as a sibling
        # of "status" -- task.message -- rather than nested inside it
        # (status.message) like every other agent tested. Check both rather
        # than needing an LLM fallback call for content that's right there.
        message_obj = status.get("message") or task.get("message") or {}
        agent_texts = _extract_texts(message_obj.get("parts", []))
        artifact_texts = []
        for artifact in task.get("artifacts") or []:
            artifact_texts.extend(_extract_texts(artifact.get("parts", [])))
        # Some agents (verified live: 2s.io) leave status.message empty and
        # put the entire answer in artifacts instead — both are spec-legal.
        agent_text = " ".join(agent_texts) or " ".join(artifact_texts)
        agent_text = await _content_or_llm_fallback(state, agent_text, artifact_texts, task)
        return {
            "task_id": task.get("id"),
            "context_id": task.get("contextId"),
            "state": state,
            "agent_text": agent_text,
            "artifacts": artifact_texts,
        }

    message = _message_from_result(result)
    if message is not None:
        agent_text = " ".join(_extract_texts(message.get("parts", [])))
        agent_text = await _content_or_llm_fallback("completed", agent_text, [], message)
        return {
            "task_id": fallback_task_id,
            "context_id": message.get("contextId", fallback_context_id),
            "state": "completed",
            "agent_text": agent_text,
            "artifacts": [],
        }

    raise UnrecognizedResponseShapeError(dialect, result)


async def _post_rpc(
    httpx_client: httpx.AsyncClient,
    rpc_url: str,
    payload: dict,
    dialect: str,
    extra_headers: dict | None = None,
) -> dict:
    """Shared POST + JSON-RPC envelope handling for any method (message
    send/stream, task get/cancel, ...). Returns the parsed `result` payload
    or raises a clean, attributed RuntimeError."""
    headers = {"A2A-Version": "1.0"} if dialect == "1.0" else {}
    if extra_headers:
        headers.update(extra_headers)
    resp = await httpx_client.post(rpc_url, json=payload, headers=headers)
    # Try to read a JSON-RPC error out of the body *before* raising on HTTP
    # status -- found live (rsperformance.online): a well-formed JSON-RPC
    # error ("-32001: Task not found or expired; IDs are short-lived...")
    # was being thrown away in favor of a generic, far-less-useful "404 Not
    # Found" from raise_for_status(), since that ran first. Its own agent
    # card explicitly documents this hybrid transport ("quick HTTP plus
    # JSON... full JSON RPC 2.0 on POST /"), so a non-2xx status carrying a
    # meaningful JSON-RPC error body is a real, intentional shape here, not
    # a malformed response -- surface the more specific error.
    try:
        body = resp.json()
    except ValueError as e:
        # Not JSON at all -- a bad HTTP status is the only signal we have,
        # so raise on that (preserves 401/402/etc classification for a
        # non-JSON-RPC-aware failure). Found live (theloopbreaker.com): a
        # target can return HTTP 200 with a non-JSON (often empty or HTML)
        # body, surfacing as a bare "Expecting value: line 1 column 1
        # (char 0)" -- accurate but gives no hint that it's *our* JSON
        # parsing of *their* response that failed.
        resp.raise_for_status()
        raise RuntimeError(f"Response body was not valid JSON: {e}") from e
    # Defensive rather than assuming a well-formed JSON-RPC envelope: found
    # live (coinrailz.com) that a target can return HTTP 200 with valid
    # JSON that has neither "result" nor "error" -- body["result"] then
    # crashed with a raw, unattributed KeyError: 'result' instead of a
    # clean protocol error. Also guards "error" being present but not a
    # dict (some agents may return a bare string/other shape there).
    if isinstance(body, dict) and "error" in body:
        error = body["error"]
        if isinstance(error, dict):
            raise RuntimeError(f"A2A error {error.get('code')}: {error.get('message')}")
        raise RuntimeError(f"A2A error: {error}")
    # No JSON-RPC error in the body -- any bad HTTP status here is a real,
    # unexplained failure (auth/payment/generic), so raise on it now.
    resp.raise_for_status()
    if "result" not in body:
        keys = sorted(body.keys()) if isinstance(body, dict) else type(body).__name__
        raise RuntimeError(f"JSON-RPC response has neither 'result' nor 'error' (keys: {keys})")
    return body["result"]


async def send_message(
    httpx_client: httpx.AsyncClient,
    rpc_url: str,
    dialect: str,
    text: str,
    task_id: str | None = None,
    context_id: str | None = None,
    extra_headers: dict | None = None,
) -> dict:
    payload = _build_request(dialect, text, task_id, context_id)
    result = await _post_rpc(httpx_client, rpc_url, payload, dialect, extra_headers)
    return await _normalize(dialect, result, task_id, context_id)


async def get_task(
    httpx_client: httpx.AsyncClient,
    rpc_url: str,
    dialect: str,
    task_id: str,
    context_id: str | None = None,
    extra_headers: dict | None = None,
) -> dict:
    """Polls `tasks/get`/`GetTask` for a task's current state -- for agents
    that reply synchronously with a non-terminal state (`working`,
    `submitted`) rather than blocking until done or requiring a follow-up
    message. Confirmed live (rsperformance.online): a real agent returns
    `working` on the *first* response and never reaches a terminal state via
    message/send alone -- the only way to see it finish is to poll the task
    by id, which prober.py's poll_task node now does instead of (wrongly)
    treating `working` like `input-required` and sending it a new chat
    message."""
    payload = _build_get_task_request(dialect, task_id)
    result = await _post_rpc(httpx_client, rpc_url, payload, dialect, extra_headers)
    return await _normalize(dialect, result, task_id, context_id)


async def check_streaming_accepted(
    httpx_client: httpx.AsyncClient, rpc_url: str, dialect: str, extra_headers: dict | None = None
) -> dict:
    """Lightweight availability check only — confirms the endpoint returns
    genuine SSE data for a streaming call, not a full event-stream consumer.

    Real streaming events are confirmed-different-shaped between dialects
    (live capture, agent.co-legal.be, v1.0-nested: `{"result":
    {"statusUpdate": {...}}}` / `{"artifactUpdate": {...}}`, each carrying
    tool-call telemetry as intermediate "working" states; live capture, our
    own fixture, v0.3-flat: `{"result": {"kind": "status-update", "final":
    bool, ...}}` / `{"kind": "artifact-update", ...}`) — genuinely new
    territory this prober doesn't consume/accumulate yet. This just answers
    "does the endpoint respond to a streaming call at all," the same
    diagnostic level as `version_probe`'s dialect-acceptance check, not a
    claim that streaming responses are parsed."""
    payload = _build_stream_request(dialect, PING_TEXT)
    headers = {"A2A-Version": "1.0"} if dialect == "1.0" else {}
    if extra_headers:
        headers.update(extra_headers)
    try:
        async with httpx_client.stream("POST", rpc_url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    return {"accepted": True, "error": None}
                if line.strip():
                    return {"accepted": False, "error": f"response did not look like SSE: {line[:200]!r}"}
        return {"accepted": False, "error": "empty response body"}
    except Exception as e:
        return {"accepted": False, "error": str(e)}


# -- Error classification/explanation, shared across every prober and
# native-client layer that speaks A2A (kept byte-identical alongside this
# file's own _normalize()/send_message(), same convention as the rest of
# this file's cross-package sharing) -----------------------------------
#
# Found live, repeatedly, across many probers this project built: a raw
# caught exception (`RuntimeError: A2A error -32601: Method not found:
# "SendMessage"...`, or some other client library's own equivalent) reached
# a customer-facing report with no indication of whether it was a bug in
# Ariel or in the target agent. version_probe.py's own _classify_error()
# already solved this for the golden probers' protocol_version_compat
# check specifically, by inspecting the *type* of exception a2a_wire.py's
# own send_message() raises -- but most of the 12 native-client layers
# raise their OWN client library's exception types, not a2a_wire.py's, so
# an isinstance-based classifier can't reach them. classify_error_text()
# is the same classification, generalized to work on the *stringified*
# error regardless of what type of exception produced it.
_ERROR_EXPLANATIONS = {
    "unsupported": (
        "This agent doesn't support the current A2A protocol version it was tested with "
        "(it only speaks an older dialect) -- no conversation could be attempted here. "
        "This is a limitation of the target agent, not a bug in this test."
    ),
    "auth_required": "This agent requires authentication that wasn't supplied -- no conversation could be attempted here.",
    "payment_required": "This agent requires payment before it will respond -- no conversation could be attempted here.",
    "internal_error": "This agent's own backend errored out while handling the request -- no conversation could be attempted here.",
}


def classify_error_text(raw: str, default: str = "other") -> str:
    """Text-only generalization of version_probe.py's own _classify_error()
    -- same substring checks, applied to the stringified error regardless
    of the exception's own type. Returns one of "unsupported" /
    "auth_required" / "payment_required" / "internal_error" / "other" (or
    `default` if nothing matches).

    `default` exists because version_probe.py's own RuntimeError branch has
    a deliberately different fallback than every other caller: ANY
    JSON-RPC-level error while probing a specific dialect is overwhelmingly
    likely to mean "this dialect isn't understood" unless it matches a more
    specific bucket first -- version_probe.py passes default="unsupported"
    to preserve that exact original behavior. Every other caller (probers
    with no dialect-specific context) wants the safer generic "other"."""
    lowered = raw.lower()
    # -32603 checked FIRST, before the looser 401/403 substring match below:
    # confirmed live (neva.dt-agent.co.uk) that a -32603 "Internal error"
    # can legitimately wrap and leak an upstream 401 INSIDE its own message
    # text ("A2A error -32603: ... {'type': 'authentication_error', ...}")
    # without meaning the caller itself needs to authenticate -- the
    # backend integration is what's broken. Checking -32603 first means
    # that case is never misclassified as auth_required just because "401"
    # happens to appear somewhere inside the wrapped text.
    if "a2a error -32603" in lowered or "-32603" in raw:
        return "internal_error"
    if "authentication required" in lowered or "401" in raw or "403" in raw:
        return "auth_required"
    if "402" in raw or "payment" in lowered:
        return "payment_required"
    if "not valid json" in lowered or "neither 'result' nor 'error'" in lowered:
        # Same reasoning as version_probe.py's own identical check: a
        # non-JSON/non-JSON-RPC-shaped body is closer to a broken/wrong
        # endpoint (404/405-flavored) than "the dialect isn't understood" --
        # always "other" regardless of `default`, a deliberate carve-out.
        return "other"
    if "method not found" in lowered or "-32601" in raw:
        return "unsupported"
    return default


def explain_error(raw: str) -> str:
    """Classifies `raw` and prepends a plain-language explanation, never
    discarding the original text -- "{explanation} (raw error: {raw})".
    Returns `raw` unchanged if it doesn't match a known bucket, so an
    error shape this project hasn't seen before is never dropped or
    garbled, just left as-is."""
    explanation = _ERROR_EXPLANATIONS.get(classify_error_text(raw))
    return f"{explanation} (raw error: {raw})" if explanation else raw
