"""Dump expected spacy.matcher.Matcher results for a curated set of
patterns × texts. Used by matcher/matcher_real_test.go for strict
100% differential.

Output: testdata/golden/matcher_cases.json with shape:

    {
      "spacy_version": "3.8.14",
      "model": "en_core_web_sm",
      "patterns": [
        {"key": "AI_PHRASE", "pattern": [{"LOWER": ...}, ...]},
        ...
      ],
      "cases": [
        {
          "text": "...",
          "matches": [{"key": "AI_PHRASE", "start": 0, "end": 2}, ...]
        },
        ...
      ]
    }

The pattern list intentionally covers every value form gospacy's
Tier-1 matcher supports — keep it expansive so adding a new value
form here is what catches a future regression. Empty `matches` is a
valid case (asserts a pattern does NOT fire).
"""

from __future__ import annotations

import json
import pathlib

import spacy
from spacy.matcher import Matcher

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "testdata" / "golden" / "matcher_cases.json"

# Each entry is (key, pattern). pattern is a list-of-list-of-dict to
# match spaCy's Matcher.add(key, [pattern1, pattern2, ...]) API.
# For Tier 1 we keep one alternative per key (the matcher engine
# already covers same-key multi-alternative via the dedup test).
PATTERNS = [
    # Scalars.
    ("APPLE_ORTH",   [[{"ORTH": "Apple"}]]),
    ("AI_LOWER",     [[{"LOWER": "ai"}]]),
    ("NNP_TAG",      [[{"TAG": "NNP"}]]),
    ("PROPN_POS",    [[{"POS": "PROPN"}]]),
    ("DOBJ_DEP",     [[{"DEP": "dobj"}]]),
    ("BUY_LEMMA",    [[{"LEMMA": "buy"}]]),
    # IN sets.
    ("AI_LOWER_IN",  [[{"LOWER": {"IN": ["artificial", "ai"]}}]]),
    ("ORG_OR_GPE",   [[{"ENT_TYPE": {"IN": ["ORG", "GPE"]}}]]),
    # NOT_IN.
    ("TAG_NOT_VBZ",  [[{"TAG": {"NOT_IN": ["VBZ", ""]}}]]),
    # REGEX on LOWER.
    ("NEG_REGEX",    [[{"LOWER": {"REGEX": "^n'?t$"}}]]),
    # IS_* flags.
    ("PUNCT_FLAG",   [[{"IS_PUNCT": True}]]),
    ("LIKE_NUM",     [[{"LIKE_NUM": True}]]),
    # Multi-token sequence.
    ("AI_PHRASE",    [[{"LOWER": "artificial"}, {"LOWER": "intelligence"}]]),
]

CASES = [
    "Apple Inc. is considering buying a U.K. startup for $1 billion.",
    "Tim Cook announced the acquisition in London on Tuesday.",
    "Artificial Intelligence will play a massive role.",
    "AI is transforming the modern tech landscape.",
    "I don't know what you mean.",            # negation contraction
    "She bought twenty apples and 3 oranges.",  # numbers
    "Hello, world!",                          # punctuation
    "He said: hello world.",                  # mixed punct
]


def main() -> int:
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    for key, pattern in PATTERNS:
        matcher.add(key, pattern)

    cases = []
    for text in CASES:
        doc = nlp(text)
        raw = matcher(doc)
        matches = []
        for match_id, start, end in raw:
            key = nlp.vocab.strings[match_id]
            matches.append({"key": key, "start": int(start), "end": int(end)})
        # Sort by (start, end, key) so the golden is stable.
        matches.sort(key=lambda m: (m["start"], m["end"], m["key"]))
        cases.append({"text": text, "matches": matches})

    payload = {
        "spacy_version": spacy.__version__,
        "model": "en_core_web_sm",
        "patterns": [{"key": k, "pattern": p} for k, p in PATTERNS],
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
