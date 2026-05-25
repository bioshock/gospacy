"""W05 — Content-word filter: lemmas of tokens where not is_stop and not is_punct.

Output: testdata/golden/workflows/w05_content_words.json
Schema: {case_id: [lemma, ...]} (token order preserved).
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[str]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    write_workflow_golden("w05_content_words", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
