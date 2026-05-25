"""W12 — Sentence count.

Output: testdata/golden/workflows/w12_sent_count.json
Schema: {case_id: int}.
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, int] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = sum(1 for _ in doc.sents)
    write_workflow_golden("w12_sent_count", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
