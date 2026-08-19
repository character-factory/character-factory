"""Schema-test fixtures: the committed example files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "characters"


@pytest.fixture(params=sorted(EXAMPLES_DIR.glob("*.char.json")), ids=lambda p: p.stem)
def example_path(request) -> Path:
    return request.param


@pytest.fixture
def example_doc(example_path: Path) -> dict:
    return json.loads(example_path.read_text(encoding="utf-8"))
