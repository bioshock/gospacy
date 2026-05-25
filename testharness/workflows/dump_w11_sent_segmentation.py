"""W11 — Sentence segmentation: text of each sentence in the doc.

Output: testdata/golden/workflows/w11_sent_segmentation.json
Schema: {case_id: [sent_text, ...]}.
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[str]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = [s.text for s in doc.sents]
    write_workflow_golden("w11_sent_segmentation", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
