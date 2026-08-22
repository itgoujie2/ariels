# Findings

This project exists because testing the A2A protocol through one client
library only tells you whether *that* library's abstraction works. These
are the real, live-confirmed gaps found by driving real, independently
discovered A2A agents through many different clients — including
several official, first-party ones. Every finding below was reproduced
against a real agent (not synthetic), and most are locked in as offline
regression tests in the corresponding package's own `tests/` directory.

None of this is a claim that any of these frameworks are bad — it's the
opposite point: even mature, official clients diverge in ways a caller
using only one of them would never see. That's exactly the value of
testing through many.

## The wire protocol itself has two live dialects

Confirmed by testing an unmodified official agent on the current
`a2a-sdk` (1.x): there are two genuinely different wire shapes in active
use, not just two client APIs.

| | v0.3 | v1.0 |
|---|---|---|
| Method | `message/send` (lowercase) | `SendMessage` (PascalCase) |
| Header | none | `A2A-Version: 1.0` required |
| Role | `"user"` / `"agent"` | `"ROLE_USER"` / `"ROLE_AGENT"` |
| Part | `{"kind": "text", "text": ...}` | `{"text": ...}` (no discriminator) |
| State | `"completed"`, `"input-required"` (kebab-case) | `"TASK_STATE_COMPLETED"`, etc. |
| Response | `{"result": {...task}}` | `{"result": {"task": {...}}}` (one level deeper) |

A client that silently negotiates down to whatever dialect a target
happens to speak (several of the clients below do exactly this) will mask
the fact that an agent doesn't actually support the current spec — which
is itself a real, reportable gap for anyone building against these agents
going forward.

## Per-client findings

What each column actually measures:

- **Reachable** — out of a private registry of 153 independently discovered
  real A2A agents (the URLs themselves stay private — see `CLAUDE.md`'s
  scope notes — but the aggregate counts below are real, live results),
  how many did this client successfully connect to and get a working
  response from at all. Only 4 of the 12 clients were run at that batch
  scale so far; `—` means "not run at that scale," not "0 passed."
- **Continuation** — when a real agent comes back mid-conversation asking
  a follow-up question (`input-required`), can the client correctly
  resume the *same* task using the real `task_id`/`context_id`, or does it
  (silently or otherwise) start an unrelated new task instead?
- **Discovery fallback** — if an agent's card isn't found at the current
  A2A spec's `.well-known` path, does the client retry the older, legacy
  discovery path, or just fail?

`—` in the Continuation/Discovery fallback columns means the writeup below
doesn't make a specific claim about that dimension for that client — not
"no issue found."

| Client | Reachable | Continuation | Discovery fallback | Key gap |
|---|---|---|---|---|
| ADK-Go | — | ✗ no plain-text resume | ✗ current path only | Rejects cards without `supportedInterfaces` |
| LangChain4j | 0/153 | — | ✗ no legacy fallback | 100% of registry unreachable |
| Mastra | — | ⚠️ fakes it | — | `resumeGenerate()` never actually continues the task |
| CrewAI | 71/153 | ⚠️ needs `task_id`, not just `context_id` | ✗ no legacy fallback | `input_required` text lands in `error`, not `result` |
| AG2 | — | ⚠️ no iteration limit | — | Blocks on real stdin unless overridden |
| fasta2a | — | — | ✗ no discovery at all | Strict validation rejects spec-legal-ish real responses |
| Agno | — | ✗ no `task_id` param | ⚠️ broken in the one working mode | Defaults to a protocol mode that 404s on non-Agno agents |
| PraisonAI | 15/153 | — | ✗ hardcoded RPC path | Wrong part discriminator (`type` vs. spec's `kind`) |
| Microsoft Agent Framework | 5/153 | ✅ best tested | — | Silently returns empty text on non-streaming `input_required` |
| Strands | — | ✗ none at all | — | Silently drops the agent's reply text entirely in a common shape |
| Raw `a2a-sdk` | — | — | — | `streaming=True` default hangs indefinitely on some real agents |
| Google ADK | — | — | — | Pinned SDK generation can't validate several real v1.0 shapes |

### ADK-Go (`google.golang.org/adk`)

The strictest client tested. Requires a card's `supportedInterfaces`
field to be present at all — a flat, legacy-style card with just a `url`
field (a large share of real agents) is rejected outright before any
request is even attempted. Has no continuation mechanism for a plain-text
`input-required` reply (only a structured "long-running tool" extension
convention) — a real multi-turn conversation with a typical agent simply
can't be resumed through this client. Also rejects several real,
standard `securitySchemes` shapes outright, and only tries the current
A2A spec's well-known discovery path with no fallback to the legacy one.

### LangChain4j (`org.a2aproject.sdk`)

Run at scale against a public registry of 153 independently discovered
A2A agents: **0 reachable.** The two dominant failure modes: "no server
interface available" (a card without `supportedInterfaces`, 69/153) and
a protobuf conversion that chokes on the card's own `security`/
`securitySchemes` field for the majority of real cards regardless of
whether a scheme is actually meaningfully declared (33/153). A further
23/153 failed specifically because the target's card was only served at
the legacy well-known path, which this client never falls back to.

### Mastra (`@mastra/core/a2a`)

The most deceptive finding of any client tested: `resumeGenerate()` does
**not** actually continue the same task, despite an API surface
(`resumePayload`, `waitingForInput`) that strongly implies it does. It
sets a `referenceTaskIds` link field but never sets the outgoing
message's own `taskId` — confirmed by checking `task.id`/`task.contextId`
directly: every "continued" turn is actually a brand-new, unrelated task.
A real multi-turn conversation driven through this client *looks*
coherent (each reply plausibly references the "previous" turn) purely
because the calling code's own follow-up text happens to restate context
— there is no real session continuity underneath, which a caller relying
on this for per-task billing or an authorization scope tied to the
original task would silently lose every single turn.

### CrewAI (`crewai.a2a`)

The richest client API of any tested, and the most permissive in
practice — 71/153 real agents reachable, the highest of any layer, with
a 96% reachable-to-passing ratio. Real findings: a task's `input_required`
clarifying text lands in the response's `error` field, not `result` (only
populated for a genuinely completed task) — a naive integration checking
only `error` to decide pass/fail would misclassify a normal mid-conversation
question as a failure. Continuing a conversation requires passing back the
*prior turn's own* `task_id`, not just `context_id` — passing only the
latter silently starts a fresh task rather than erroring. Card discovery
only tries the current well-known path, no legacy fallback. A real agent
declaring 120 skills crashed the client's own internal event-stack-depth
limit (100) — a resource limit with nothing to do with the wire protocol.

### AG2 (`autogen.a2a`)

Its internal `input-required` handling loop has **no iteration limit at
all** — the only way to break it is answering the literal string
`"exit"`, a hardcoded convention in the client's own source, with no
documented bound otherwise. Defaults to blocking on real stdin input
unless the extension point is overridden. Loses a real agent's actual
clarifying-question text when the agent puts it in `status.message`
rather than a separate `history` array (confirmed against a real agent) —
falls back to a generic placeholder, discarding the real question
silently. Also drops any `DataPart` content from a response, keeping only
`TextPart`.

### fasta2a (pydantic-ai)

The only client tested with zero dependency on `a2a-sdk` — a from-scratch
implementation. Speaks v0.3 only, with no `.well-known` discovery of its
own at all (a caller must already know the RPC endpoint). Strict pydantic
response validation rejects at least two real, live agents' response
shapes that are spec-legal-ish but non-conformant (a missing `kind`
discriminator; a history message missing several expected fields) —
shapes other, more tolerant clients parse without incident.

### Agno (formerly Phidata)

Defaults to a `protocol="rest"` mode that constructs a URL specific to
Agno's own server (`{base_url}/v1/message:send`) — a clean 404 against
any real, non-Agno agent; a caller must already know to pass
`protocol="json-rpc"` explicitly. Its own `get_agent_card()` is broken
specifically in the one mode that works against real agents (ignores the
path argument, re-requests the RPC endpoint instead of the well-known
path). v0.3 only. No `task_id` parameter anywhere in its public
`send_message()` signature — continuing a conversation with only
`context_id` silently starts a new task each time, with no way to pass
the real ID even if a caller wanted to. Never reads artifact text into
its own `content` field — only artifact metadata.

### PraisonAI

The most restrictive client tested by a wide margin — 15/153 reachable.
The decisive finding: its outgoing message parts use `{"type": "text"}`
as the discriminator instead of the spec's `{"kind": "text"}` — a
standards-conformant agent rejects the request outright
(`-32602 Invalid params`) before any real logic runs. This is a bug in
the client's own *outgoing* request construction, the only finding of
that shape across every client tested (every other finding here
is about response parsing). Also hardcodes its RPC path to `{base_url}/a2a`
regardless of what a target's card actually declares, and — when that
guessed path resolves to something that isn't an A2A endpoint at all
(a normal website's own 404 page) — its error handling doesn't check
whether the response body is JSON before using it as the error message,
so a real HTML error page can end up as the entire "error" text.

### Microsoft Agent Framework

The most spec-correct client tested in terms of continuation mechanics
(explicitly tracks `context_id`/`task_id`/`task_state` and threads both
back correctly) — but its default, non-streaming call silently returns
**empty text** whenever the response lands in a non-terminal state
delivered as a single-shot response, confirmed against a real agent
mid-conversation: a legitimate, substantive clarifying question came back
as an empty string while the session's own tracked state correctly showed
`input_required`. The identical call with streaming enabled returns the
real text. Also the strictest on dialect currency of any client tested —
5/153 reachable, since it refuses any dialect but the current v1.0 with
zero fallback — which is arguably *correct*, closely matching this
project's own probers' "always latest, no compromise" policy.

### Strands (AWS)

Provides **no continuation mechanism whatsoever** for a multi-turn
conversation — every call builds a fresh message with no `task_id`/
`context_id`, and documented `**kwargs` for this are explicitly ignored.
More seriously: its response conversion silently drops the agent's reply
text entirely for a common, real response shape (a completed task
delivered as a plain tuple with no separate update event) — a caller
using the public, documented API would get back a completely empty reply
despite the agent answering correctly. The same conversion also loses the
real task state in that shape, collapsing success/failure/rejection into
one indistinguishable signal.

### Raw `a2a-sdk` usage

The closest thing to a "no framework" baseline. Defaults to
`streaming=True`, which genuinely hangs against real agents that don't
handle streaming well (confirmed against a real target: instant response
with `streaming=False` set explicitly, indefinite hang without it).
Auto-negotiates wire dialect from the target's own card — the opposite of
this project's own probers' deliberate policy, and a real illustration of
why that policy exists: a client that defers to a target's self-declared
version will silently mask exactly the kind of version gap this project
is built to surface.

### Google ADK (`RemoteA2aAgent`)

The official first-party ADK client. Its pinned `a2a-sdk` generation
can't validate several real v1.0 response shapes (no top-level `url`
field, a nested result shape) that a more tolerant parser handles fine —
confirmed via direct comparison, some real agents are reachable through
this client and not others, and vice versa, purely due to SDK-generation
differences rather than anything about the agents themselves.

## Running these checks yourself

Every finding above is reproducible with the corresponding package's own
CLI — see the root [README](README.md) for how to run any of them against
a live agent, and each package's own module docstrings for the full
detail behind each finding summarized here.
