#!/usr/bin/env python3
"""Dump spaCy lex_attr_getter outputs for fixed words.

Output: testdata/golden/lex_attrs.json
Schema: {"words": [{"text": str, "prefix": str, "suffix": str, "shape": str,
                    "is_alpha": bool, "is_digit": bool, "is_punct": bool,
                    "is_space": bool, "is_lower": bool, "is_upper": bool,
                    "is_title": bool, "is_ascii": bool}]}
"""

import json
import sys
from pathlib import Path

import spacy
from spacy.lang.en import English

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"

WORDS = [
    "Hello", "hello", "HELLO", "world", "1999",
    "3.14", ",", ".", "!", "  ",
    "Don't", "U.S.A.", "spaCy", "ünïcødé", "—",
    "ab", "x", "X", "AbCd", "1a2b",
]


def main() -> int:
    nlp = English()
    out = {"words": []}
    for w in WORDS:
        lex = nlp.vocab[w]
        out["words"].append({
            "text": w,
            "prefix": lex.prefix_,
            "suffix": lex.suffix_,
            "shape": lex.shape_,
            "is_alpha": bool(lex.is_alpha),
            "is_digit": bool(lex.is_digit),
            "is_punct": bool(lex.is_punct),
            "is_space": bool(lex.is_space),
            "is_lower": bool(lex.is_lower),
            "is_upper": bool(lex.is_upper),
            "is_title": bool(lex.is_title),
            "is_ascii": bool(lex.is_ascii),
        })
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / "lex_attrs.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
