"""LLM glue for the generic prober.

Per PLAN.md §5.2, the Protocol Conformance Layer has zero LLM *judgment* — but
a prober that must work against an arbitrary, previously-unseen customer agent
still needs *something* to decide what to say next, since it can't know the
target's domain in advance. The two calls here generate plausible user turns
only; nothing in this module ever grades a response. See assertions.py for
the actual pass/fail logic, which never looks at LLM output quality.
"""

from __future__ import annotations

import asyncio
import json
import os
import random

import anthropic
from anthropic import AsyncAnthropic

MODEL = "claude-haiku-4-5-20251001"

_client: AsyncAnthropic | None = None

# Retried in _complete() with our own explicit exponential backoff --
# these are transient, infra-level failures (the target genuinely might
# succeed on a later attempt). 4xx errors (auth, bad request, etc.) are
# deliberately excluded and propagate immediately -- no amount of
# waiting fixes a bad API key.
#
# Built via getattr rather than direct attribute access: this file is
# kept byte-identical across 14 engine directories (see CLAUDE.md), each
# with its own independently-resolved venv -- confirmed live that one of
# them (agent-framework-client-checks, constrained to an older anthropic
# SDK by agent-framework-anthropic's own pin) lacks OverloadedError
# entirely, which a hardcoded tuple would crash on at import time.
# Degrading gracefully to whatever exception classes a given SDK
# version actually has is safer than forcing every engine's own
# independently-resolved dependency set to match.
_RETRYABLE_ERROR_NAMES = (
    "OverloadedError",
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
)
_RETRYABLE_ERRORS = tuple(
    cls for cls in (getattr(anthropic, name, None) for name in _RETRYABLE_ERROR_NAMES) if cls is not None
)
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 1.0


def _get_client() -> AsyncAnthropic:
    # Lazy on purpose: a2a_wire.py imports this module unconditionally, and
    # the offline test suite must be able to import it (and even call
    # _normalize()) without ANTHROPIC_API_KEY set, as long as it never
    # actually needs to make a call.
    global _client
    if _client is None:
        # max_retries=0: our own explicit retry loop in _complete() below
        # handles backoff -- disabling the SDK's own opaque internal
        # retries (default 2) keeps attempt counts/timing deterministic
        # and testable at our layer instead of split across two
        # independent, uncoordinated retry mechanisms.
        _client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)
    return _client


async def _complete(prompt: str, max_tokens: int = 60) -> str:
    """Retries transient failures (529 Overloaded, 429 rate limit, 5xx,
    connection/timeout errors) with exponential backoff + jitter before
    giving up. Found live via product/'s orchestrator
    (insideout.luthersystems.com): a transient Anthropic 529 crashed the
    whole engine subprocess -- more likely than it sounds once 13
    engines fire concurrent LLM calls at once against the same target.
    Non-retryable errors (auth, bad request, etc.) and final exhaustion
    both propagate to the caller -- generate_probe/generate_followup/
    extract_content each have their own fallback for that (see their
    docstrings), since this text is never judged (module docstring
    above) and a total/persistent LLM outage still shouldn't crash the
    actual A2A protocol test."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = await _get_client().messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # Found live (rettfrabonden.com, via the content-extraction
            # fallback): an empty response.content crashed
            # resp.content[0] with a raw, unattributed 'list index out
            # of range' several layers away from here. Not retried --
            # this isn't a transient infra failure, it's a shape the
            # model itself returned.
            if not resp.content:
                raise RuntimeError("LLM response had no content blocks")
            return resp.content[0].text.strip().strip('"')
        except _RETRYABLE_ERRORS as e:
            last_exc = e
            if attempt == _MAX_ATTEMPTS - 1:
                break
            delay = _BASE_DELAY_S * (2**attempt) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
    raise last_exc


async def generate_probe(skill: dict) -> str:
    """Synthesize a first user message for a skill the agent card gave no example for."""
    prompt = (
        "You are simulating a real end user of an AI agent, for interoperability testing.\n"
        f"Skill name: {skill['name']}\n"
        f"Skill description: {skill['description']}\n\n"
        "Write ONE short, realistic first message a user might send to invoke this skill. "
        "Reply with only the message text, no quotes, no explanation."
    )
    try:
        return await _complete(prompt)
    except Exception:
        # This text is never judged (see module docstring) -- a generic
        # fallback keeps the actual A2A protocol test running instead of
        # failing the whole probe over an unrelated, exhausted-retries
        # LLM outage (see _complete's docstring for the real incident
        # this guards against).
        return f"Can you help me with {skill['name']}?"


async def extract_content(raw_obj: dict) -> str:
    """Fallback content-location for response shapes deterministic field-
    checking doesn't recognize. Real agents keep inventing new places to put
    their answer (status.message, artifacts, data parts, and whatever comes
    next) -- rather than keep hardcoding a field list and patching it every
    time we find another real agent doing something new, ask the model to
    find it wherever it is. Only called when deterministic extraction
    (a2a_wire._extract_texts) comes up completely empty despite a
    'completed' state -- see a2a_wire._content_or_llm_fallback."""
    prompt = (
        "This is a raw JSON object from an A2A protocol agent response, "
        "already confirmed to be a completed task or message. Automated "
        "field parsing found no recognizable text content in the usual "
        "places. Find the substantive answer/content a human would care "
        "about, wherever in this JSON it's nested. Reply with ONLY the "
        "extracted content as plain text (no commentary, no JSON), or an "
        "empty string if there truly is no real content.\n\n"
        f"{json.dumps(raw_obj, indent=2)[:4000]}"
    )
    try:
        return await _complete(prompt, max_tokens=300)
    except Exception:
        # Same fallback rationale as generate_probe/generate_followup --
        # this is itself already the last-resort content-extraction path
        # (see a2a_wire._content_or_llm_fallback), so on failure there's
        # nothing left to fall back to but empty content, which callers
        # already handle as a legitimate "no content found" outcome.
        return ""


async def generate_followup(agent_prompt: str, skill: dict) -> str:
    """Synthesize a follow-up reply when the agent asked for more info (input-required)."""
    prompt = (
        "You are continuing a conversation with an AI agent, playing a realistic user, "
        "for interoperability testing.\n"
        f"The agent's skill is: {skill['name']} — {skill['description']}\n"
        f'The agent just said: "{agent_prompt}"\n\n'
        "Write ONE short, realistic reply that supplies whatever the agent is asking for "
        "so the conversation can move toward completion. If it needs an identifier you "
        "don't have, invent a plausible-looking one. Reply with only the message text, "
        "no quotes, no explanation."
    )
    try:
        return await _complete(prompt)
    except Exception:
        return "Yes, please continue — here's what you need: test-12345"
