#!/usr/bin/env python3
"""AI-Judgment Layer CLI (PLAN.md §5.6): point it at ANY A2A agent's base URL.

    python run_judgment.py http://127.0.0.1:9100

Deliberately a SEPARATE entry point from run_prober.py, not a flag on it --
this is an explicit opt-in add-on (multi-run, multi-turn, LLM-judged), never
invoked by reports/'s batch pipeline or product/'s existing 13-engine registry,
both of which shell out to run_prober.py <url> today with no flags. Output is
always clearly marked "probabilistic": true and is never combined into a
single pass/fail with the deterministic prober's own report -- see
app/judgment.py's module docstring for why.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import httpx  # noqa: E402

from app import a2a_wire, judgment  # noqa: E402

LATEST_DIALECT = "1.0"


async def main(
    base_url: str,
    extra_headers: dict | None = None,
    runs: int = judgment.DEFAULT_RUNS,
    tolerance: float = judgment.DEFAULT_TOLERANCE,
    only_skill: str | None = None,
) -> dict:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as httpx_client:
        try:
            card = await a2a_wire.fetch_card(httpx_client, base_url)
        except Exception as e:
            return {
                "target": base_url,
                "layer": "ai_judgment_probabilistic",
                "probabilistic": True,
                "error": a2a_wire.explain_error(f"{type(e).__name__}: {e}"),
            }
        rpc_url, declared_dialect = a2a_wire.resolve_target(card, base_url)

        skills = a2a_wire.get_skills(card)
        if not skills:
            skills = [
                {
                    "id": "__generic__",
                    "name": card.get("name", "generic"),
                    "description": card.get("description") or "General-purpose agent, no declared skills.",
                    "examples": [],
                }
            ]
        if only_skill:
            skills = [s for s in skills if s["id"] == only_skill]

        skill_results = []
        for skill in skills:
            result = await judgment.run_judgment_for_skill(
                httpx_client, rpc_url, LATEST_DIALECT, skill, extra_headers, runs=runs, tolerance=tolerance
            )
            skill_results.append(result)

        return {
            "target": base_url,
            "rpc_url": rpc_url,
            "declared_dialect": declared_dialect,
            "tested_dialect": LATEST_DIALECT,
            "agent_name": card.get("name"),
            "layer": "ai_judgment_probabilistic",
            "probabilistic": True,
            "runs_per_skill": runs,
            "tolerance": tolerance,
            "skills_judged": skill_results,
        }


RUNS_DIR = Path(__file__).parent / "judgment_runs"


def _write_run(base_url: str, result: dict) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_url).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RUNS_DIR / f"{timestamp}_{slug}.json"
    out_path.write_text(json.dumps(result, indent=2))
    (RUNS_DIR / "latest.json").write_text(json.dumps(result, indent=2))
    return out_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-Judgment Layer CLI")
    parser.add_argument("base_url")
    parser.add_argument("--header", action="append", default=[], help='e.g. --header "Name: Value"')
    parser.add_argument("--runs", type=int, default=judgment.DEFAULT_RUNS)
    parser.add_argument("--tolerance", type=float, default=judgment.DEFAULT_TOLERANCE)
    parser.add_argument("--skill", default=None, help="Judge only this skill id, not every declared skill.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    headers: dict = {}
    for h in args.header:
        name, _, value = h.partition(":")
        headers[name.strip()] = value.strip()

    result = asyncio.run(main(args.base_url, headers, runs=args.runs, tolerance=args.tolerance, only_skill=args.skill))
    out_path = _write_run(args.base_url, result)
    print(json.dumps(result, indent=2))
    print(f"\nsaved to {out_path}", file=sys.stderr)
    sys.exit(0 if "error" not in result else 1)
