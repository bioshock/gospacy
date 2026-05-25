#!/usr/bin/env python3
"""Generate the PrecomputableAffine.v1 golden fixture.

Builds a thinc PrecomputableAffine with hand-set W/b/pad, runs forward on a
fixed (T, nI) input, and dumps {input, W, b, pad, output} for the Go-side
parity test.
"""

import json
from pathlib import Path

import numpy as np
from spacy.ml._precomputable_affine import PrecomputableAffine

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"


def arr(a: np.ndarray) -> dict:
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": a.flatten().tolist()}


def main() -> int:
    rng = np.random.default_rng(0)
    T, nI, nO, nF, nP = 3, 4, 5, 8, 2
    X = rng.standard_normal((T, nI)).astype(np.float32)
    W = (rng.standard_normal((nF, nO, nP, nI)).astype(np.float32) * 0.1)
    b = (rng.standard_normal((nO, nP)).astype(np.float32) * 0.1)
    pad = (rng.standard_normal((1, nF, nO, nP)).astype(np.float32) * 0.1)

    model = PrecomputableAffine(nO=nO, nI=nI, nF=nF, nP=nP)
    model.initialize(X=X)
    model.set_param("W", W)
    model.set_param("b", b)
    model.set_param("pad", pad)

    Yf, _ = model(X, is_train=False)
    # Yf has shape (T+1, nF, nO, nP); flatten to inspect from Go.

    GOLDEN.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "spacy.PrecomputableAffine.v1 forward, T=3, nI=4, nO=5, nF=8, nP=2",
        "dims": {"T": T, "nI": nI, "nO": nO, "nF": nF, "nP": nP},
        "input": arr(X),
        "W": arr(W),
        "b": arr(b),
        "pad": arr(pad),
        "output": arr(Yf),
    }
    (GOLDEN / "precomputable_affine.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {GOLDEN / 'precomputable_affine.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
