"""W14 — Past-tense verb extraction: POS == VERB AND morph contains Tense=Past.

Output: testdata/golden/workflows/w14_past_tense.json
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
            if t.pos_ == "VERB" and t.morph.get("Tense") == ["Past"]
        ]
    write_workflow_golden("w14_past_tense", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
