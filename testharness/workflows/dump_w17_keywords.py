"""W17 — Keyword extraction: lemma freq for non-stop NOUN/PROPN tokens.

Output: testdata/golden/workflows/w17_keywords.json
Schema: {case_id: {lemma: count, ...}}.
"""

from __future__ import annotations

import sys
from collections import Counter

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, dict[str, int]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        counts: Counter[str] = Counter(
            t.lemma_
            for t in doc
            if t.pos_ in ("NOUN", "PROPN") and not t.is_stop
        )
        out[c["id"]] = dict(counts)
    write_workflow_golden("w17_keywords", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
