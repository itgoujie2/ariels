#!/usr/bin/env python3
"""CLI: python run_check.py <base_url>"""
from __future__ import annotations

import asyncio
import json
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

from adk_check import check  # noqa: E402


async def main(base_url: str) -> dict:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        result = await check(base_url, client)
    return {"target": base_url, "adk_native_client": result}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run_check.py <base_url>", file=sys.stderr)
        sys.exit(1)
    result = asyncio.run(main(sys.argv[1]))
    print(json.dumps(result, indent=2))
