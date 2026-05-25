"""W15 — Noun chunks: spaCy's syntax_iterators.noun_chunks output.

Output: testdata/golden/workflows/w15_noun_chunks.json
Schema: {case_id: [{start, end, text}, ...]} half-open [start,end) token
ranges (start = chunk.start, end = chunk.end).
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[dict]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = [
            {"start": ch.start, "end": ch.end, "text": ch.text}
            for ch in doc.noun_chunks
        ]
    write_workflow_golden("w15_noun_chunks", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
