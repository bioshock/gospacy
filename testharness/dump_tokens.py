#!/usr/bin/env python3
"""Dump tokenizer output: per-sentence list of [orth, idx, whitespace] tuples.

Output: testdata/golden/tokens-<corpus>.json
Schema: {"corpus": str, "spacy_version": str, "model": str,
         "sentences": [{"text": str, "tokens": [{"orth": str, "idx": int, "ws": bool}]}]}
"""

import sys
from pathlib import Path

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
        doc = nlp.tokenizer(text)  # tokenizer-only; avoids running full pipeline
        out["sentences"].append({
            "text": text,
            "tokens": [
                {"orth": t.text, "idx": t.idx, "ws": bool(t.whitespace_)}
                for t in doc
            ],
        })
    write_json(GOLDEN / f"tokens-{corpus_name}.json", out)
    return 0


if __name__ == "__main__":
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else "tokenizer-10k"
    sys.exit(dump(corpus_name))
