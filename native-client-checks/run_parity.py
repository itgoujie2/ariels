#!/usr/bin/env python3
"""Full skill-level parity check: point it at ANY A2A agent's base URL, same
as the golden probers' run_prober.py, but calling out through ADK's real
official A2A client (RemoteA2aAgent) instead of our hand-rolled a2a_wire.py
transport -- see native_client_prober.py's docstring for the design
rationale and PLAN.md §11a for why this second layer exists at all.

Usage: python run_parity.py <base_url>

Card discovery/skill enumeration uses our own tolerant a2a_wire.fetch_card()
purely to learn what skills exist and what to ask about -- a real developer
would read the card themselves to know this too. The actual test --
whether a real ADK caller can complete each skill -- goes entirely through
RemoteA2aAgent, including letting IT do its own (separate, often stricter)
card resolution and rejecting an agent outright if that fails, exactly as
a real caller relying solely on RemoteA2aAgent would experience.
"""
from __future__ import annotations

import asyncio
import json
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

import re
from datetime import datetime, timezone
from pathlib import Path

from app import a2a_wire, assertions, report_render  # noqa: E402
from native_client_prober import probe_skill_native  # noqa: E402

CARD_PATHS = a2a_wire.CARD_PATHS


async def main(base_url: str) -> dict:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as httpx_client:
        try:
            card = await a2a_wire.fetch_card(httpx_client, base_url)
        except Exception as e:
            return {
                "target": base_url,
                "overall_pass": False,
                "error": a2a_wire.explain_error(f"{type(e).__name__}: {e}"),
            }

        skills = a2a_wire.get_skills(card)
        if not skills:
            skills = [{
                "id": "__generic__", "name": card.get("name", "generic"),
                "description": card.get("description") or "General-purpose agent, no declared skills.",
                "examples": [],
            }]

        # RemoteA2aAgent does its own card resolution independently, and
        # lazily -- construction alone never raises; resolution only
        # happens on the first real run_async() call. So each candidate
        # path has to actually be exercised with a real ping, not just
        # constructed, to find out which one (if any) really works for it.
        from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
        from google.adk.runners import InMemoryRunner
        from google.genai import types as genai_types

        agent_card_url = None
        last_attempt_error = None
        for path in CARD_PATHS:
            candidate = base_url.rstrip("/") + path
            attempt_error = None
            try:
                probe_agent = RemoteA2aAgent(name="probe_setup", agent_card=candidate, httpx_client=httpx_client)
                runner = InMemoryRunner(agent=probe_agent)
                session = await runner.session_service.create_session(
                    app_name=runner.app_name, user_id="probe_setup"
                )
                async for event in runner.run_async(
                    user_id="probe_setup", session_id=session.id,
                    new_message=genai_types.Content(role="user", parts=[genai_types.Part(text="ping")]),
                ):
                    if event.error_message:
                        attempt_error = a2a_wire.explain_error(event.error_message)
            except Exception as e:
                attempt_error = a2a_wire.explain_error(f"{type(e).__name__}: {e}")

            if attempt_error is None:
                agent_card_url = candidate
                last_attempt_error = None
                break
            last_attempt_error = attempt_error

        report: dict = {
            "target": base_url,
            "agent_name": card.get("name"),
            "declared_skills": [s["id"] for s in skills],
            "prober_implementation": "google-adk-native-client",
        }

        if agent_card_url is None:
            report["overall_pass"] = False
            report["error"] = last_attempt_error
            report["skills_tested"] = []
            return report

        results = []
        for skill in skills:
            transcript = await probe_skill_native(agent_card_url, skill, httpx_client)
            results.append(assertions.summarize(skill["id"], transcript))

        report["skills_tested"] = results
        report["overall_pass"] = all(r["pass"] for r in results)
        return report


RUNS_DIR = Path(__file__).parent / "runs"


def _write_run(base_url: str, result: dict) -> Path:
    """A standalone, self-contained HTML report alongside the raw JSON --
    see app/report_render.py's own docstring for why this exists: a
    customer running this open-source layer directly, with no hosted
    product involved, should still get something to actually look at."""
    RUNS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_url).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RUNS_DIR / f"{timestamp}_{slug}.json"
    out_path.write_text(json.dumps(result, indent=2))
    (RUNS_DIR / "latest.json").write_text(json.dumps(result, indent=2))
    html = report_render.render(result, engine_label="Google ADK native client")
    (RUNS_DIR / f"{timestamp}_{slug}.html").write_text(html)
    (RUNS_DIR / "latest.html").write_text(html)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run_parity.py <base_url>", file=sys.stderr)
        sys.exit(1)
    result = asyncio.run(main(sys.argv[1]))
    _write_run(sys.argv[1], result)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("overall_pass") else 1)
