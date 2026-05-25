#!/usr/bin/env python3
"""Dump per-token tags, fine POS, morph, and lemmas after running the FULL pipeline
on each input sentence (so tagger + attribute_ruler + lemmatizer have all run).

Output: testdata/golden/attrs-<corpus>.json
Schema: {"corpus", "spacy_version", "model", "pipeline": [str],
         "sentences": [{"text", "tokens": [{"orth", "tag", "pos", "morph", "lemma"}]}]}
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
        "pipeline": list(nlp.pipe_names),
        "sentences": [],
    }
    for text in iter_corpus_lines(corpus):
        doc = nlp(text)
        out["sentences"].append({
            "text": text,
            "tokens": [
                {
                    "orth": t.text,
                    "tag": t.tag_,
                    "pos": t.pos_,
                    "morph": str(t.morph),
                    "lemma": t.lemma_,
                }
                for t in doc
            ],
        })
    write_json(GOLDEN / f"attrs-{corpus_name}.json", out)
    return 0


if __name__ == "__main__":
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else "tokenizer-10k"
    sys.exit(dump(corpus_name))
