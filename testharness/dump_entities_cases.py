#!/usr/bin/env python3
"""Dump per-token EntIOB + EntType for the shared pipeline cases.

Output: testdata/golden/entities_cases{_md,_lg}.json — keyed by case id, each
value is a list of {text, ent_iob, ent_type} dicts in token order.

ent_iob is the IOB letter ("B", "I", "O"); at inference, spaCy collapses
its internal L codes to "I" via Doc.set_ents, so we record the same.
ent_type is the label string (e.g. "PERSON", "ORG"; empty for OUT tokens).

Honours GOSPACY_MODEL (defaults to en_core_web_sm). md/lg suffixes match
common.py's golden_suffix().
"""

from __future__ import annotations

import json
import pathlib
import sys

from common import GOLDEN, MODEL_NAME, golden_suffix, load_nlp

REPO = pathlib.Path(__file__).resolve().parents[1]
CASES = REPO / "testharness" / "pipeline_cases.json"
OUT = GOLDEN / f"entities_cases{golden_suffix()}.json"


def main() -> int:
    nlp = load_nlp()
    with CASES.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    out: dict[str, list[dict]] = {}
    for c in cases:
        doc = nlp(c["text"])
        out[c["id"]] = [
            {
                "text": t.text,
                "ent_iob": t.ent_iob_,    # "B", "I", "O" — spaCy collapses L→I at inference
                "ent_type": t.ent_type_,  # label string, empty for OUT
            } for t in doc
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT.relative_to(REPO)} ({len(out)} cases, model={MODEL_NAME})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
