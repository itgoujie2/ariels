#!/usr/bin/env python3
"""Batch-run the full skill-level native-client parity check (run_parity.py)
across a target list.

Usage: python batch_parity.py <targets_file> [output_name]
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from run_parity import main as parity_main  # noqa: E402

CONCURRENCY = 4  # lower than the shallow check's 6 -- each agent may drive several skills x several turns
REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "reports"


async def probe_one(sem: asyncio.Semaphore, target: str) -> dict:
    async with sem:
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(parity_main(target), timeout=180)
        except Exception as e:
            result = {"target": target, "overall_pass": False, "error": f"{type(e).__name__}: {e}"}
        result["_elapsed_s"] = round(time.monotonic() - start, 1)
        return result


async def run(targets: list[str], output_name: str) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [asyncio.create_task(probe_one(sem, t)) for t in targets]
    all_results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        all_results.append(result)
        status = "PASS" if result.get("overall_pass") else "FAIL"
        print(f"[{i}/{len(targets)}] {status} {result['target']}", file=sys.stderr)
    (REPORTS_DIR / f"{output_name}.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    targets_file = sys.argv[1] if len(sys.argv) > 1 else "targets.txt"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "ag2_parity_results"
    targets = [line.strip() for line in Path(targets_file).read_text().splitlines() if line.strip()]
    print(f"checking {len(targets)} targets with concurrency={CONCURRENCY}...", file=sys.stderr)
    asyncio.run(run(targets, output_name))
