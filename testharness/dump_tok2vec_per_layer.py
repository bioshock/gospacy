#!/usr/bin/env python3
"""Dump per-layer tok2vec outputs for a fixed sentence from en_core_web_sm.

Captures the activations at the 6 architectural boundaries:
  - after ExtractFeatures (List[Ints2d])
  - after List2Ragged (Ragged)
  - after MultiHashEmbed (Ragged of concatenated embeddings)
  - after embed-reduce (Ragged of width-96)
  - after Ragged2List (List[Floats2d])
  - after MaxoutWindowEncoder (List[Floats2d])  ← final tok2vec output
"""

import json
from pathlib import Path

import numpy as np
import spacy

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"

TEXT = "The quick brown fox jumps over the lazy dog."


def array_to_json(arr) -> dict:
    arr = np.asarray(arr)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "data": arr.flatten().tolist()}


def main() -> int:
    nlp = spacy.load("en_core_web_sm")
    tok2vec_pipe = nlp.get_pipe("tok2vec")
    model = tok2vec_pipe.model
    doc = nlp.make_doc(TEXT)

    # Inspected real-bundle structure (see plan Step 1 NOTE):
    #   model.layers[0] = chain(extract_features, list2ragged, with_array(concat),
    #                            with_array(reduce-maxout), ragged2list)   — 5 children
    #   model.layers[1] = with_array(residual(EW+max+LN+drop) × 4)         — the encoder
    extract_features = model.layers[0].layers[0]
    list2ragged = model.layers[0].layers[1]
    multi_hash_embed = model.layers[0].layers[2]
    embed_reduce = model.layers[0].layers[3]
    ragged2list = model.layers[0].layers[4]
    encode = model.layers[1]

    out = {"text": TEXT, "n_tokens": len(doc), "boundaries": {}}

    feats = extract_features.predict([doc])
    out["boundaries"]["extract_features"] = [array_to_json(f) for f in feats]

    ragged = list2ragged.predict(feats)
    out["boundaries"]["list2ragged"] = {
        "data": array_to_json(ragged.data),
        "lengths": ragged.lengths.tolist(),
    }

    embedded = multi_hash_embed.predict(ragged)
    out["boundaries"]["multi_hash_embed"] = {
        "data": array_to_json(embedded.data),
        "lengths": embedded.lengths.tolist(),
    }

    reduced = embed_reduce.predict(embedded)
    out["boundaries"]["embed_reduce"] = {
        "data": array_to_json(reduced.data),
        "lengths": reduced.lengths.tolist(),
    }

    encoded_in = ragged2list.predict(reduced)
    out["boundaries"]["ragged2list"] = [array_to_json(item) for item in encoded_in]

    encoded = encode.predict(encoded_in)
    out["boundaries"]["encode"] = [array_to_json(item) for item in encoded]

    final = model.predict([doc])
    out["final"] = [array_to_json(item) for item in final]

    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "tok2vec_per_layer.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {GOLDEN / 'tok2vec_per_layer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
