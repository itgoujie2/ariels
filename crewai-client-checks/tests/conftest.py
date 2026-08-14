import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture():
    def _load(name: str) -> dict:
        with open(FIXTURES_DIR / f"{name}.json") as f:
            body = json.load(f)
        return body["result"]

    return _load
