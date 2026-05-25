"""Dump per-token lemma for the shared pipeline cases.

Output: testdata/golden/lemmatizer{_suffix}.json — keyed by case id.

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
OUT = REPO / "testdata" / "golden" / f"lemmatizer{golden_suffix()}.json"


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"model not present at {MODEL_PATH}; run make download-assets", file=sys.stderr)
        return 1
    nlp = spacy.load(str(MODEL_PATH))
    with CASES.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    out: dict[str, list[dict]] = {}
    for c in cases:
        doc = nlp(c["text"])
        out[c["id"]] = [
            {"text": t.text, "lemma": t.lemma_, "pos": t.pos_} for t in doc
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT.relative_to(REPO)} ({len(out)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
