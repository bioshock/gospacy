"""W03 — Comparative/superlative tags (JJR/JJS/RBR/RBS).

Output: testdata/golden/workflows/w03_comparatives.json
Schema: {case_id: [{text, tag}, ...]} (token order preserved).
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden

_TARGET_TAGS = {"JJR", "JJS", "RBR", "RBS"}


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[dict[str, str]]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = [
            {"text": t.text, "tag": t.tag_} for t in doc if t.tag_ in _TARGET_TAGS
        ]
    write_workflow_golden("w03_comparatives", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
