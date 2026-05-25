#!/usr/bin/env python3
"""Generate the FeatureExtractor.v1 golden fixture.

Builds a real en_core_web_sm Doc, runs FeatureExtractor on it with the same
6-column attribute list the bundle uses (NORM,PREFIX,SUFFIX,SHAPE,SPACY,IS_SPACE),
and dumps the resulting Uint64s2d for the Go-side test.
"""

import json
from pathlib import Path

import numpy as np
import spacy
from spacy.attrs import IS_SPACE, NORM, PREFIX, SHAPE, SPACY, SUFFIX
from spacy.ml.featureextractor import FeatureExtractor

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"

CASES = ["Hello world.", "The cat sat."]


def main() -> int:
    nlp = spacy.load("en_core_web_sm")
    columns = [NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE]
    column_names = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"]
    fx = FeatureExtractor(columns)
    fx.initialize()

    docs = [nlp.make_doc(text) for text in CASES]
    out_list, _ = fx(docs, is_train=False)

    out_payload = []
    for text, arr in zip(CASES, out_list):
        out_payload.append({
            "text": text,
            "shape": list(arr.shape),
            "data": arr.flatten().tolist(),
            "tokens": [{"text": t.text, "whitespace": t.whitespace_, "shape": t.shape_} for t in nlp.make_doc(text)],
        })

    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "extract_features.json").write_text(json.dumps({
        "columns": column_names,
        "docs": out_payload,
    }, indent=2))
    print(f"wrote {GOLDEN / 'extract_features.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
