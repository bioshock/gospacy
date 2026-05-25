"""W16 — Question detection: per sentence, returns True if either
(a) the sentence's first token's tag_ starts with "W" (WDT, WP, WP$, WRB),
or (b) the sentence ends with '?'.

Output: testdata/golden/workflows/w16_questions.json
Schema: {case_id: [bool, ...]} one bool per sentence in order.
"""

from __future__ import annotations

import sys

from _common import load_cases, load_nlp, write_workflow_golden


def _is_question(sent) -> bool:
    if len(sent) == 0:
        return False
    first = sent[0]
    if first.tag_.startswith("W"):
        return True
    last = sent[-1]
    if last.text == "?":
        return True
    return False


def main() -> int:
    nlp = load_nlp()
    out: dict[str, list[bool]] = {}
    for c in load_cases():
        doc = nlp(c["text"])
        out[c["id"]] = [_is_question(s) for s in doc.sents]
    write_workflow_golden("w16_questions", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
