"""W09 — Passive-voice detection: tokens with dep_ == nsubjpass.

Output: testdata/golden/workflows/w09_passive.json
Schema: {case_id: [{text, head}, ...]} (head = text of the parent token).
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[dict[str, str]]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = [
            {"text": t.text, "head": t.head.text}
            for t in doc
            if t.dep_ == "nsubjpass"
        ]
    write_workflow_golden("w09_passive", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
