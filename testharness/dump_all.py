#!/usr/bin/env python3
"""Regenerate all golden fixtures.

Runs all dump_*.py scripts in sequence against the default corpus, plus a
tiny in-repo "sample" corpus used by Go unit tests (so Go tests don't need
the 10k corpus to be present)."""

import json
import subprocess
import sys
from pathlib import Path

import spacy

from common import GOLDEN, REPO, load_nlp, write_json

CORPUS_FULL = "tokenizer-10k"
SAMPLE_SENTENCES = [
    "Apple is looking at buying U.K. startup for $1 billion.",
    "The quick brown fox jumps over the lazy dog.",
    "I do n't think she'd come tomorrow.",
]


def dump_sample() -> None:
    """Produce small in-repo fixtures from SAMPLE_SENTENCES — used by Go unit tests."""
    nlp = load_nlp()
    tokens_out = {"spacy_version": spacy.__version__, "model": "en_core_web_sm", "sentences": []}
    attrs_out = {"spacy_version": spacy.__version__, "model": "en_core_web_sm",
                 "pipeline": list(nlp.pipe_names), "sentences": []}
    arcs_out = {"spacy_version": spacy.__version__, "model": "en_core_web_sm", "sentences": []}
    ents_out = {"spacy_version": spacy.__version__, "model": "en_core_web_sm", "sentences": []}
    for text in SAMPLE_SENTENCES:
        doc_tok = nlp.tokenizer(text)
        tokens_out["sentences"].append({
            "text": text,
            "tokens": [{"orth": t.text, "idx": t.idx, "ws": bool(t.whitespace_)} for t in doc_tok],
        })
        doc = nlp(text)
        attrs_out["sentences"].append({
            "text": text,
            "tokens": [{"orth": t.text, "tag": t.tag_, "pos": t.pos_,
                        "morph": str(t.morph), "lemma": t.lemma_} for t in doc],
        })
        arcs_out["sentences"].append({
            "text": text,
            "arcs": [{"i": t.i, "head": t.head.i, "dep": t.dep_} for t in doc],
        })
        ents_out["sentences"].append({
            "text": text,
            "entities": [{"start": e.start, "end": e.end, "label": e.label_, "text": e.text}
                         for e in doc.ents],
        })
    write_json(GOLDEN / "sample_tokens.json", tokens_out)
    write_json(GOLDEN / "sample_attrs.json", attrs_out)
    write_json(GOLDEN / "sample_arcs.json", arcs_out)
    write_json(GOLDEN / "sample_entities.json", ents_out)
    # Also persist the input sentences so Go tests can iterate them.
    write_json(GOLDEN / "sample_input.json", {"sentences": SAMPLE_SENTENCES})


def main() -> int:
    here = REPO / "testharness"
    py = sys.executable
    for script in ("dump_tokens.py", "dump_attrs.py", "dump_arcs.py", "dump_entities.py"):
        print(f"--- running {script} on {CORPUS_FULL} ---")
        rc = subprocess.call([py, str(here / script), CORPUS_FULL])
        if rc != 0:
            return rc
    print("--- generating in-repo sample fixtures ---")
    dump_sample()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
