#!/usr/bin/env python3
"""Batch-run the CrewAI client compatibility check across a target list.

Usage: python batch_check.py <targets_file> [output_name]

Mirrors the shape of the probers' own batch_probe.py, but this is a much
lighter check (one message via CrewAI's own first-party A2A client) run
as a supplementary pass over the same target list, not a replacement for
the main probers' full skill-by-skill conformance run.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from crewai_check import check  # noqa: E402

CONCURRENCY = 6
REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "reports"


async def check_one(sem: asyncio.Semaphore, target: str) -> dict:
    async with sem:
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(check(target), timeout=60)
        except Exception as e:
            result = {"reachable": False, "reply": None, "error": f"{type(e).__name__}: {e}"}
        return {"target": target, "crewai_client": result, "_elapsed_s": round(time.monotonic() - start, 1)}


async def run(targets: list[str], output_name: str) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [asyncio.create_task(check_one(sem, t)) for t in targets]
    all_results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        all_results.append(result)
        status = "REACHABLE" if result["crewai_client"]["reachable"] else "UNREACHABLE"
        print(f"[{i}/{len(targets)}] {status} {result['target']}", file=sys.stderr)
    (REPORTS_DIR / f"{output_name}.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    targets_file = sys.argv[1] if len(sys.argv) > 1 else "targets.txt"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "crewai_client_results"
    targets = [line.strip() for line in Path(targets_file).read_text().splitlines() if line.strip()]
    print(f"checking {len(targets)} targets with concurrency={CONCURRENCY}...", file=sys.stderr)
    asyncio.run(run(targets, output_name))
