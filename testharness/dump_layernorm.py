#!/usr/bin/env python3
"""Generate the LayerNorm.v1 golden fixture.

Builds a thinc LayerNorm with hand-set G/b, runs forward on a fixed input,
dumps {input, G, b, output} for the Go-side parity test.
"""

import json
from pathlib import Path

import numpy as np
from thinc.api import LayerNorm

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"


def array_to_json(arr: np.ndarray) -> dict:
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "data": arr.flatten().tolist()}


def main() -> int:
    rng = np.random.default_rng(0)
    N, nI = 5, 8
    X = rng.standard_normal((N, nI)).astype(np.float32)
    G = (rng.standard_normal((nI,)).astype(np.float32) * 0.5) + 1.0
    b = rng.standard_normal((nI,)).astype(np.float32) * 0.1

    model = LayerNorm(nI=nI)
    model.initialize(X=X)
    model.set_param("G", G)
    model.set_param("b", b)

    Y, _ = model(X, is_train=False)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "thinc LayerNorm.v1 forward, N=5, nI=8, eps=1e-8",
        "dims": {"N": N, "nI": nI},
        "input": array_to_json(X),
        "G": array_to_json(G),
        "b": array_to_json(b),
        "output": array_to_json(Y),
    }
    (GOLDEN / "layernorm.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {GOLDEN / 'layernorm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
