"""W01 — POS-tag frequency table per case.

Output: testdata/golden/workflows/w01_pos_freq.json
Schema: {case_id: {POS: count, ...}, ...}
"""

from __future__ import annotations

import sys
from collections import Counter

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    cases = load_cases()
    out: dict[str, dict[str, int]] = {}
    for c in cases:
        doc = nlp(c["text"])
        counts: Counter[str] = Counter(t.pos_ for t in doc)
        out[c["id"]] = dict(counts)
    write_workflow_golden("w01_pos_freq", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
