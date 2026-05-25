#!/usr/bin/env python3
"""Generate the Dropout.v1 inference-identity golden fixture."""

import json
from pathlib import Path

import numpy as np
from thinc.api import Dropout

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"


def array_to_json(arr: np.ndarray) -> dict:
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "data": arr.flatten().tolist()}


def main() -> int:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 6)).astype(np.float32)

    model = Dropout(rate=0.1)
    # Inference: is_train=False — should pass through.
    Y, _ = model(X, is_train=False)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "thinc Dropout.v1 inference (is_train=False) — identity",
        "rate": 0.1,
        "is_enabled": True,
        "input": array_to_json(X),
        "output": array_to_json(Y),
    }
    (GOLDEN / "dropout.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {GOLDEN / 'dropout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
