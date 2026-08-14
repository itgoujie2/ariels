#!/usr/bin/env python3
"""Full skill-level parity check: point it at ANY A2A agent's base URL,
same as the golden probers' run_prober.py, but calling out through
pydantic-ai's own first-party `fasta2a.client.A2AClient` instead of our
hand-rolled `a2a_wire.py` transport -- see fasta2a_prober.py's docstring
for the design rationale and PLAN.md §11a for why this layer exists.

Usage: python run_parity.py <base_url>
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
from fasta2a_prober import _explain_layer_error, probe_skill_fasta2a, resolve_client  # noqa: E402


async def main(base_url: str) -> dict:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as httpx_client:
        try:
            card = await a2a_wire.fetch_card(httpx_client, base_url)
        except Exception as e:
            return {
                "target": base_url,
                "overall_pass": False,
                "error": _explain_layer_error(f"{type(e).__name__}: {e}"),
            }

        skills = a2a_wire.get_skills(card)
        if not skills:
            skills = [{
                "id": "__generic__", "name": card.get("name", "generic"),
                "description": card.get("description") or "General-purpose agent, no declared skills.",
                "examples": [],
            }]

        client, _card, resolve_error = await resolve_client(base_url, httpx_client)

        report: dict = {
            "target": base_url,
            "agent_name": card.get("name"),
            "declared_skills": [s["id"] for s in skills],
            "prober_implementation": "fasta2a-client",
        }

        if client is None:
            report["overall_pass"] = False
            report["error"] = resolve_error
            report["skills_tested"] = []
            return report

        results = []
        for skill in skills:
            transcript = await probe_skill_fasta2a(client, skill)
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
    html = report_render.render(result, engine_label="fasta2a native client")
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
