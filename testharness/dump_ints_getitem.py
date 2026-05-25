#!/usr/bin/env python3
"""Generate the ints_getitem column-slice golden fixture."""

import json
from pathlib import Path

import numpy as np
from thinc.layers.array_getitem import ints_getitem

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"


def main() -> int:
    rng = np.random.default_rng(2)
    N, K = 4, 6
    X = rng.integers(0, 1_000_000, size=(N, K), dtype=np.uint64)

    # column = 3 — pulls X[:, 3] as Ints1d.
    model = ints_getitem((slice(0, None), 3))
    Y, _ = model(X, is_train=False)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "thinc ints_getitem((slice(0, None), 3)) over uint64 (N, K)",
        "dims": {"N": N, "K": K, "col": 3},
        "input": {"shape": list(X.shape), "data": X.flatten().tolist()},
        "output": {"shape": list(Y.shape), "data": Y.flatten().tolist()},
    }
    (GOLDEN / "ints_getitem.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {GOLDEN / 'ints_getitem.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
