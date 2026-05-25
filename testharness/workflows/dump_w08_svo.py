"""W08 — SVO triples (nsubj + verb head + dobj).

For each verb V whose children include both an nsubj S and a dobj O,
emit (S.text, V.text, O.text). Skips passive constructions.

Output: testdata/golden/workflows/w08_svo.json
Schema: {case_id: [[subj, verb, obj], ...]}.
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[list[str]]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        triples: list[list[str]] = []
        for tok in doc:
            if tok.pos_ != "VERB":
                continue
            subj = None
            obj = None
            for child in tok.children:
                if child.dep_ == "nsubj" and subj is None:
                    subj = child.text
                elif child.dep_ == "dobj" and obj is None:
                    obj = child.text
            if subj is not None and obj is not None:
                triples.append([subj, tok.text, obj])
        out[c["id"]] = triples
    write_workflow_golden("w08_svo", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
