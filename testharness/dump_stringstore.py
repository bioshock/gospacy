#!/usr/bin/env python3
"""Dump Python spaCy StringStore hashes for a fixed set of strings.

Output: testdata/golden/stringstore.json
Schema: {"strings": [{"text": str, "hash": str (decimal uint64)}]}
"""

import json
import sys
from pathlib import Path

from spacy.strings import StringStore

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"

CASES = [
    "",
    "a",
    "the",
    "spaCy",
    "hello world",
    "ünïcødé",
    "VERB",
    "NOUN",
    "Det",
    "the quick brown fox jumps over the lazy dog",
    "a" * 32,
    "—",  # em dash
    "'s",
    "n't",
]


def main() -> int:
    s = StringStore()
    out = {"strings": []}
    for text in CASES:
        h = s.add(text) if text else 0
        out["strings"].append({"text": text, "hash": str(h)})
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / "stringstore.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
