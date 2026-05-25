"""W04 — Unique lemma set per case (sorted, alphabetical).

Output: testdata/golden/workflows/w04_unique_lemmas.json
Schema: {case_id: [lemma, ...]} (sorted alphabetical).
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[str]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = sorted({t.lemma_ for t in doc})
    write_workflow_golden("w04_unique_lemmas", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
