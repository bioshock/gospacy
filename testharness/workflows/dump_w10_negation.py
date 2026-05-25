"""W10 — Negation scope: for each token whose dep_ == neg, emit
{neg_text, head_text, head_pos}. spaCy attaches "neg" to the head it
negates (commonly a verb).

Output: testdata/golden/workflows/w10_negation.json
Schema: {case_id: [{neg, head, head_pos}, ...]}.
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
            {"neg": t.text, "head": t.head.text, "head_pos": t.head.pos_}
            for t in doc
            if t.dep_ == "neg"
        ]
    write_workflow_golden("w10_negation", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
