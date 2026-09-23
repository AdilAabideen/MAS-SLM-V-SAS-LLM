"""Fixtures shared by research and backend tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures"
REPO_ROOT = TESTS_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def load_json_fixture():
    """Read a checked-in provider or golden-trace JSON fixture."""

    def _load(relative_path: str) -> dict | list:
        return json.loads((FIXTURES_ROOT / relative_path).read_text())

    return _load
