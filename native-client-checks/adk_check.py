"""Native-client compatibility check using Google ADK's own, official,
currently-shipped A2A client (`RemoteA2aAgent`, the `google-adk[a2a]` extra).

Distinct from `reference-agents/ours/adk-prober/`, which hand-rolls the wire
protocol (`a2a_wire.py`) to test whether the *target* speaks the current A2A
wire dialect correctly, independent of any calling library's own bugs. This
module asks a different, complementary question instead: can a *real*
production caller -- one built by simply following Google's own official
docs (`pip install google-adk[a2a]`) -- actually reach and use this agent at
all?

The two questions genuinely diverge. Confirmed live: `api.moltrust.ch` is
fully spec-compliant per the ground-truth prober, but totally unreachable
through `RemoteA2aAgent`, because `google-adk[a2a]`'s pinned dependency
(`a2a-sdk>=0.3.4,<0.4`) requires a card's top-level `url` field that no
current, correctly-built v1.0-only card has (they declare
`supportedInterfaces[].url` instead). A customer's agent can be perfectly
protocol-compliant and still be unreachable by real callers built on a
popular framework's own official, currently-shipped integration -- that gap
is the whole point of this check.

Deliberately minimal: does the client resolve the card, and does sending one
basic message get a real reply? Not a full skill-by-skill probe like the
main prober -- RemoteA2aAgent's own maturity (it ships marked
[EXPERIMENTAL] by Google itself) makes a lightweight reachability check the
honest scope for a first version.
"""

from __future__ import annotations

import httpx
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

CARD_PATHS = ["/.well-known/agent-card.json", "/.well-known/agent.json"]
PROBE_TEXT = "What can you help me with? Please describe your capabilities."


async def check(base_url: str, httpx_client: httpx.AsyncClient) -> dict:
    """Returns {"reachable": bool, "reply": str | None, "error": str | None}.

    Tries both known card discovery paths (same fallback the main prober's
    fetch_card() uses) before giving up.
    """
    last_error: str | None = None
    for path in CARD_PATHS:
        card_url = base_url.rstrip("/") + path
        try:
            agent = RemoteA2aAgent(name="native_check", agent_card=card_url, httpx_client=httpx_client)
            runner = InMemoryRunner(agent=agent)
            session = await runner.session_service.create_session(
                app_name=runner.app_name, user_id="native_check"
            )
            reply_text = None
            error_text = None
            async for event in runner.run_async(
                user_id="native_check",
                session_id=session.id,
                new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=PROBE_TEXT)]),
            ):
                if event.error_message:
                    error_text = event.error_message
                if event.content and event.content.parts:
                    texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
                    if texts:
                        reply_text = " ".join(texts)
            if error_text:
                last_error = error_text
                continue
            return {"reachable": True, "reply": reply_text, "error": None}
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
    return {"reachable": False, "reply": None, "error": last_error}
