#!/usr/bin/env python3
"""Dump dependency parse + POS for the shared pipeline cases.

Output: testdata/golden/parser_arcs{_suffix}.json — keyed by case id, each
value is a list of {i, text, head, dep, pos} dicts in token order.

`head == i` denotes a root. `pos` is included so the parser-real-test can
verify the AR re-classifies s07 tokens 3 and 5 correctly once Token.Dep is
populated (closing KNOWN_DIVERGENCES.md s07 POS gap).

Set GOSPACY_MODEL=en_core_web_md (or en_core_web_lg) for md/lg variants.
"""

from __future__ import annotations

import json
import pathlib
import sys

import spacy

from common import MODEL_PATH, golden_suffix

REPO = pathlib.Path(__file__).resolve().parents[1]
CASES = REPO / "testharness" / "pipeline_cases.json"
OUT = REPO / "testdata" / "golden" / f"parser_arcs{golden_suffix()}.json"


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"model not present at {MODEL_PATH}; run make download-assets", file=sys.stderr)
        return 1
    nlp = spacy.load(str(MODEL_PATH))
    with CASES.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    out = {}
    for c in cases:
        doc = nlp(c["text"])
        out[c["id"]] = [
            {"i": t.i, "text": t.text, "head": t.head.i, "dep": t.dep_, "pos": t.pos_}
            for t in doc
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT.relative_to(REPO)} ({len(out)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
