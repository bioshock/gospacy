#!/usr/bin/env python3
"""Dump NER entity spans.

Output: testdata/golden/entities-<corpus>.json
Schema: {"corpus", "spacy_version", "model",
         "sentences": [{"text", "entities": [{"start": int, "end": int,
                                              "label": str, "text": str}]}]}

start/end are token indices (Span.start, Span.end).
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
            "entities": [
                {"start": ent.start, "end": ent.end, "label": ent.label_, "text": ent.text}
                for ent in doc.ents
            ],
        })
    write_json(GOLDEN / f"entities-{corpus_name}.json", out)
    return 0


if __name__ == "__main__":
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else "tokenizer-10k"
    sys.exit(dump(corpus_name))
