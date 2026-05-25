"""W02 — Proper-noun text list per case (POS == PROPN).

Output: testdata/golden/workflows/w02_propn.json
Schema: {case_id: [text, ...]} (token order preserved).
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[str]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = [t.text for t in doc if t.pos_ == "PROPN"]
    write_workflow_golden("w02_propn", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
