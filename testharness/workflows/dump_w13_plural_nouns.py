"""W13 — Plural-noun extraction: POS == NOUN AND morph contains Number=Plur.

Output: testdata/golden/workflows/w13_plural_nouns.json
Schema: {case_id: [text, ...]}.
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[str]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = [
            t.text
            for t in doc
            if t.pos_ == "NOUN" and t.morph.get("Number") == ["Plur"]
        ]
    write_workflow_golden("w13_plural_nouns", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
