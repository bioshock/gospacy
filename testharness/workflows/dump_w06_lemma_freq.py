"""W06 — Lemma frequency excluding stop words and punctuation.

Output: testdata/golden/workflows/w06_lemma_freq.json
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
            t.lemma_ for t in doc if not t.is_stop and not t.is_punct
        )
        out[c["id"]] = dict(counts)
    write_workflow_golden("w06_lemma_freq", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
