#!/usr/bin/env python3
"""Generate the HashEmbed-with-column golden fixture.

We DO NOT exercise thinc's chain(ints_getitem, hashembed) here — that's
Block C's job (the rebuilt MultiHashEmbed factory). We only need to prove the
LEAF HashEmbed still produces the same output for a given Ints1d input, AND
that the column attr is round-trippable.
"""

import json
from pathlib import Path

import numpy as np
from thinc.api import HashEmbed

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"


def main() -> int:
    rng = np.random.default_rng(4)
    nO, nV, column, seed = 8, 50, 2, 10
    E = rng.standard_normal((nV, nO)).astype(np.float32)
    ids = rng.integers(0, 1_000_000, size=(7,), dtype=np.uint64)

    model = HashEmbed(nO=nO, nV=nV, column=column, seed=seed, dropout=0.1)
    # The chain wraps (ints_getitem, hashembed); ints_getitem expects 2-D input.
    # Use a 2-D array for initialization; the leaf forward takes 1-D (Ints1d).
    ids2d = np.stack([ids] * (column + 1), axis=1)  # shape (7, column+1)
    model.initialize(X=ids2d)
    # The HashEmbed factory wraps via chain(ints_getitem, hashembed) when
    # column != None; pull the inner leaf for parameter setting.
    leaf = model.layers[1]
    leaf.set_param("E", E)

    # Forward is_train=False through the *leaf* directly on Ints1d to match the
    # gospacy code path; the chain wrapper is Block C territory.
    Y, _ = leaf(ids, is_train=False)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "thinc HashEmbed leaf with column=2, seed=10, nO=8, nV=50",
        "dims": {"nO": nO, "nV": nV},
        "attrs": {"column": column, "seed": seed, "dropout_rate": 0.1},
        "E": {"shape": [nV, nO], "data": E.flatten().tolist()},
        "ids": ids.tolist(),
        "output": {"shape": list(Y.shape), "data": Y.flatten().tolist()},
    }
    (GOLDEN / "hashembed_column.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {GOLDEN / 'hashembed_column.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
