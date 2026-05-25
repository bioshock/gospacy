"""W07 — Sentence ROOT tokens: per case, list every token whose dep_ == ROOT.

Output: testdata/golden/workflows/w07_sent_roots.json
Schema: {case_id: [{text, dep, pos}, ...]} one entry per ROOT token in
token order.

Note: this is the dep-label-based definition of "root". For our 8 fixture
sentences (which are all single-clause) it coincides with sentence.root,
but it doesn't require Span.root semantics — only Dep + POS.
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
            {"text": t.text, "dep": t.dep_, "pos": t.pos_}
            for t in doc
            if t.dep_ == "ROOT"
        ]
    write_workflow_golden("w07_sent_roots", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
