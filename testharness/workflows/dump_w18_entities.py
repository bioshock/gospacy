"""Workflow W18: entity extraction. Per-doc list of {start, end, label, text}.

Mirrors:
    [{"start": e.start, "end": e.end, "label": e.label_, "text": e.text}
     for e in doc.ents]
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
            {"start": e.start, "end": e.end, "label": e.label_, "text": e.text}
            for e in doc.ents
        ]
    write_workflow_golden("w18_entities", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
