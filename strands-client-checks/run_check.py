#!/usr/bin/env python3
"""CLI: python run_check.py <base_url>"""
from __future__ import annotations

import asyncio
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from strands_check import check  # noqa: E402


async def main(base_url: str) -> dict:
    result = await check(base_url)
    return {"target": base_url, "strands_client": result}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run_check.py <base_url>", file=sys.stderr)
        sys.exit(1)
    result = asyncio.run(main(sys.argv[1]))
    print(json.dumps(result, indent=2))
