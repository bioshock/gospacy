#!/usr/bin/env python3
"""Dump dependency parse arcs.

Output: testdata/golden/arcs-<corpus>.json
Schema: {"corpus", "spacy_version", "model",
         "sentences": [{"text", "arcs": [{"i": int, "head": int, "dep": str}]}]}

Note: `i` and `head` are token indices within the doc. `head == i` means root.
"""

import sys

import spacy

from common import CORPORA, GOLDEN, iter_corpus_lines, load_nlp, write_json


def dump(corpus_name: str) -> int:
    corpus = CORPORA / f"{corpus_name}.txt"
    if not corpus.exists():
        print(f"missing corpus: {corpus}", file=sys.stderr)
        return 1
    nlp = load_nlp()
    out = {
        "corpus": corpus_name,
        "spacy_version": spacy.__version__,
        "model": "en_core_web_sm",
        "sentences": [],
    }
    for text in iter_corpus_lines(corpus):
        doc = nlp(text)
        out["sentences"].append({
            "text": text,
            "arcs": [
                {"i": t.i, "head": t.head.i, "dep": t.dep_}
                for t in doc
            ],
        })
    write_json(GOLDEN / f"arcs-{corpus_name}.json", out)
    return 0


if __name__ == "__main__":
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else "tokenizer-10k"
    sys.exit(dump(corpus_name))
