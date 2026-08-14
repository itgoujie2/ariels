#!/usr/bin/env python3
"""Batch re-probe a list of targets with the CURRENT prober code, for report
accuracy -- some targets in reports/ were originally captured against
earlier, since-fixed code mid-session, so their recorded pass/fail no longer
reflects what the tool (or the agent) actually does today. Writes one JSON
result per target to reports/batch/, plus a combined reports/batch_results.json.

Usage: python batch_probe.py ../../reports/targets.txt [output_name]

`output_name` (default "batch_results") controls both the per-target
subdirectory under reports/batch/<output_name>/ and the combined
reports/<output_name>.json -- pass a distinct name for a new batch so it
doesn't clobber an existing combined results file (e.g. the one backing
the published sales report).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from run_prober import main as probe_main  # noqa: E402

CONCURRENCY = 6
REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "reports"


async def probe_one(sem: asyncio.Semaphore, target: str) -> dict:
    async with sem:
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(probe_main(target), timeout=120)
        except Exception as e:
            result = {"target": target, "overall_pass": False, "error": f"{type(e).__name__}: {e}"}
        result["_probe_elapsed_s"] = round(time.monotonic() - start, 1)
        return result


async def run(targets: list[str], output_name: str) -> None:
    out_dir = REPORTS_DIR / "batch" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [asyncio.create_task(probe_one(sem, t)) for t in targets]
    all_results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        all_results.append(result)
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", result["target"]).strip("-")
        (out_dir / f"{slug}.json").write_text(json.dumps(result, indent=2))
        status = "PASS" if result.get("overall_pass") else "FAIL"
        print(f"[{i}/{len(targets)}] {status} {result['target']}", file=sys.stderr)
    (REPORTS_DIR / f"{output_name}.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    targets_file = sys.argv[1] if len(sys.argv) > 1 else "targets.txt"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "batch_results"
    targets = [line.strip() for line in Path(targets_file).read_text().splitlines() if line.strip()]
    print(f"probing {len(targets)} targets with concurrency={CONCURRENCY}...", file=sys.stderr)
    asyncio.run(run(targets, output_name))
