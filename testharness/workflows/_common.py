"""Shared helpers for workflow dump scripts.

testharness/workflows/ lives one directory below testharness/, so import
the existing common module via an explicit sys.path insert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TESTHARNESS = HERE.parent
REPO = TESTHARNESS.parent
sys.path.insert(0, str(TESTHARNESS))

# Re-export from the parent common module so callers can import from one place.
from common import GOLDEN, load_nlp  # noqa: E402

CASES_PATH = TESTHARNESS / "pipeline_cases.json"
GOLDEN_DIR = GOLDEN / "workflows"


def load_cases() -> list[dict[str, str]]:
    with CASES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def write_workflow_golden(name: str, payload: dict[str, Any]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLDEN_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
