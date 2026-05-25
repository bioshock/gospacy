#!/usr/bin/env python3
"""Dump tokenizer output for a fixed set of test strings.

Output: testdata/golden/tokenizer_cases.json
Schema: {"cases": [{"text": str,
                    "tokens": [{"orth": str, "idx": int, "ws": bool}]}]}
"""

import json
import sys
from pathlib import Path

from spacy.lang.en import English

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"

CASES = [
    "",
    "hello",
    "hello world",
    "Hello, world!",
    "Don't go.",
    "He said \"yes\".",
    "U.S.A. is large.",
    "I went to the U.S.",
    "Visit https://example.com today.",
    "(hello)",
    "1+1=2",
    "$50",
    "50%",
    "What?!",
    "well-known issue",
    "co-author",
    "1999-2024",
    "Mr. Smith arrived.",
    "I.B.M. and A.I.",
    "It's a test, isn't it?",
    "Email: me@example.com.",
    "—",
    "...",
    "    leading spaces",
    "trailing spaces    ",
    "multiple   spaces",
    "\tTabbed\t",
    "ünïcødé tëst",
    "1. First\n2. Second",
    "no-pre—em—dash-fix",
]


def main() -> int:
    nlp = English()
    out = {"cases": []}
    for text in CASES:
        doc = nlp.tokenizer(text)
        out["cases"].append({
            "text": text,
            "tokens": [
                {"orth": t.text, "idx": t.idx, "ws": bool(t.whitespace_)}
                for t in doc
            ],
        })
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / "tokenizer_cases.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
