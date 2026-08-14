#!/usr/bin/env python3
"""Generic prober CLI: point it at ANY A2A agent's base URL.

    python run_prober.py http://127.0.0.1:9100

Google ADK port of `reference-agents/ours/langgraph-prober/run_prober.py` --
identical behavior and identical report shape, ported per PLAN.md §11 to
prove the same testing capability is portable across agent frameworks
(the orchestration engine is ADK's BaseAgent/Runner here instead of
LangGraph's StateGraph; see app/prober.py's docstring for why).

Always probes using the latest A2A wire dialect (v1.0), regardless of what
the card declares. An agent that only supports an older dialect will fail
every skill probe as a direct, intended consequence — that's a real finding
to report to the customer ("your agent doesn't support the current A2A
protocol version"), not something this tool works around by matching
whatever version the target happens to speak. `protocol_version_compat`
(from version_probe.py) still tests every known dialect for diagnostic
context, explaining *why* the skill probes failed if the target is behind.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import httpx  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

from app import a2a_wire, assertions, report_render, version_probe  # noqa: E402
from app.prober import ProberAgent  # noqa: E402

LATEST_DIALECT = "1.0"

SELF_DESCRIBE_SKILL = {
    "id": "__self_describe__",
    "name": "Self-description",
    "description": "Ask the agent directly what it can help with.",
    "examples": ["What can you help me with? Please describe your capabilities."],
}


async def probe_skill(
    httpx_client: httpx.AsyncClient,
    rpc_url: str,
    dialect: str,
    skill: dict,
    extra_headers: dict | None = None,
) -> dict:
    agent = ProberAgent(
        name="prober",
        skill=skill,
        httpx_client=httpx_client,
        rpc_url=rpc_url,
        dialect=dialect,
        extra_headers=extra_headers or {},
    )
    runner = InMemoryRunner(agent=agent)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="ariel")
    async for _ in runner.run_async(
        user_id="ariel",
        session_id=session.id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text="probe")]),
    ):
        pass
    return agent.result


async def main(base_url: str, extra_headers: dict | None = None) -> dict:
    # follow_redirects=True: found live (agent.moirailabs.com) that a card
    # can declare a plain-http RPC url that the server itself 301s to
    # https -- the same host/path, just a scheme upgrade. Every real HTTP
    # client (browsers, curl -L, virtually every SDK) follows that
    # transparently; refusing to would make this a stricter, less
    # representative test of real interoperability than any actual
    # counterparty agent would perform.
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as httpx_client:
        try:
            card = await a2a_wire.fetch_card(httpx_client, base_url)
        except Exception as e:
            # Undiscoverable is a distinct, top-level failure -- report it
            # cleanly rather than crashing with a raw traceback. Confirmed
            # live (insideout.luthersystems.com): a misconfigured discovery
            # endpoint should surface as "couldn't find this agent," not a
            # pile of confusing downstream errors from a bogus empty card.
            return {
                "target": base_url,
                "overall_pass": False,
                "error": a2a_wire.explain_error(f"{type(e).__name__}: {e}"),
            }
        rpc_url, declared_dialect = a2a_wire.resolve_target(card, base_url)

        report: dict = {
            "target": base_url,
            "rpc_url": rpc_url,
            "declared_dialect": declared_dialect,  # what the card claims to speak
            "tested_dialect": LATEST_DIALECT,  # what we actually always probe with
            "agent_name": card.get("name"),
            "agent_description": card.get("description"),
            "declared_skills": [s["id"] for s in a2a_wire.get_skills(card)],
            "prober_implementation": "google-adk",
        }

        report["protocol_version_compat"] = await version_probe.probe(
            httpx_client, card, rpc_url, extra_headers
        )
        latest_result = report["protocol_version_compat"]["dialects_tested"].get(LATEST_DIALECT, {})
        supports_latest = latest_result.get("accepted", False)
        report["supports_latest_a2a_version"] = supports_latest
        if latest_result.get("reason") == "auth_required":
            # Distinct from "doesn't support the dialect" -- the server
            # understood the request fine and is gating on credentials we
            # don't have. Reporting this as a version-support failure would
            # be a real, misleading error, not just an imprecise one.
            report["note"] = (
                f"This agent requires authentication for v{LATEST_DIALECT} calls and none was "
                "supplied (use --header to pass credentials). Protocol-version support could not "
                "be confirmed either way — this is an auth gap, not a 'wrong A2A version' finding."
            )
        elif latest_result.get("reason") == "internal_error":
            # Distinct again: the request was understood fine, but something
            # broke on the target's own backend while handling it (confirmed
            # live: neva.dt-agent.co.uk leaking an upstream API-key failure
            # through a JSON-RPC -32603). Not a protocol-version problem.
            report["note"] = (
                f"This agent's own backend errored out while handling the v{LATEST_DIALECT} call "
                "(JSON-RPC -32603 Internal error) — protocol-version support could not be confirmed "
                "either way. This looks like an outage or misconfiguration on the target's side, not "
                "a 'wrong A2A version' finding."
            )
        elif latest_result.get("reason") == "payment_required":
            # Same principle again, different gate: understood fine, wants
            # payment (x402) rather than auth. Confirmed live: gpt55.558686.xyz.
            report["note"] = (
                f"This agent requires payment (HTTP 402) for v{LATEST_DIALECT} calls — protocol-"
                "version support could not be confirmed either way. This is a payment gap, not a "
                "'wrong A2A version' finding."
            )
        elif not supports_latest:
            report["note"] = (
                f"This agent does not accept the current A2A wire dialect (v{LATEST_DIALECT}) "
                "at all. Every skill probe below is expected to fail as a direct consequence — "
                "see protocol_version_compat for what it does accept."
            )

        self_desc = await probe_skill(
            httpx_client, rpc_url, LATEST_DIALECT, SELF_DESCRIBE_SKILL, extra_headers
        )
        report["live_self_description"] = self_desc.get("final_message")

        skills = a2a_wire.get_skills(card)
        if not skills:
            # Card declares nothing — fall back to one generic probe so we
            # still exercise *something* rather than skipping the target.
            skills = [
                {
                    "id": "__generic__",
                    "name": card.get("name", "generic"),
                    "description": card.get("description") or "General-purpose agent, no declared skills.",
                    "examples": [],
                }
            ]

        results = []
        for skill in skills:
            transcript = await probe_skill(httpx_client, rpc_url, LATEST_DIALECT, skill, extra_headers)
            results.append(assertions.summarize(skill["id"], transcript))

        report["skills_tested"] = results
        # Not supporting the latest dialect at all is a real, reportable
        # failure now — that's the point (see module docstring), not
        # something this tool works around.
        report["overall_pass"] = all(r["pass"] for r in results) and supports_latest
        return report


RUNS_DIR = Path(__file__).parent / "runs"


def _write_run(base_url: str, result: dict) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_url).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RUNS_DIR / f"{timestamp}_{slug}.json"
    out_path.write_text(json.dumps(result, indent=2))
    (RUNS_DIR / "latest.json").write_text(json.dumps(result, indent=2))
    # A standalone, self-contained HTML report alongside the raw JSON --
    # see app/report_render.py's own docstring for why this exists.
    html = report_render.render(result, engine_label="ADK golden prober")
    (RUNS_DIR / f"{timestamp}_{slug}.html").write_text(html)
    (RUNS_DIR / "latest.html").write_text(html)
    return out_path


def _parse_args(argv: list[str]) -> tuple[str, dict]:
    if not argv:
        print('usage: python run_prober.py <base_url> [--header "Name: Value"]...', file=sys.stderr)
        sys.exit(1)
    base_url = argv[0]
    headers: dict = {}
    i = 1
    while i < len(argv):
        if argv[i] == "--header":
            name, _, value = argv[i + 1].partition(":")
            headers[name.strip()] = value.strip()
            i += 2
        else:
            i += 1
    return base_url, headers


if __name__ == "__main__":
    base_url, extra_headers = _parse_args(sys.argv[1:])
    result = asyncio.run(main(base_url, extra_headers))
    out_path = _write_run(base_url, result)
    print(json.dumps(result, indent=2))
    print(f"\nsaved trajectory to {out_path}", file=sys.stderr)
    sys.exit(0 if result["overall_pass"] else 1)
