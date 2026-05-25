#!/usr/bin/env python3
"""Build a tiny thinc model, save its bytes, and dump per-layer expected outputs.

Outputs:
  testdata/golden/tiny_thinc_model.msgpack   — model.to_bytes() raw
  testdata/golden/tiny_thinc_model_io.json   — input + expected outputs per layer

The Go side hand-constructs the matching tree, loads the .msgpack via
nn.FromBytes, runs forward on the dumped input, and compares per-layer
outputs at every level of the tree.
"""

import json
import sys
from pathlib import Path

import numpy as np
import thinc
from thinc.api import Linear, Softmax_v2, chain

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"
SEED = 42

# Model: Linear(3 → 4) → Softmax(4 → 2). Inputs are (batch, 3) Floats2d.
N_IN, N_HIDDEN, N_OUT = 3, 4, 2


def array_to_json(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.flatten().tolist(),
    }


def main() -> int:
    rng = np.random.default_rng(SEED)
    GOLDEN.mkdir(parents=True, exist_ok=True)

    # Build the model.
    linear = Linear(nO=N_HIDDEN, nI=N_IN)
    softmax = Softmax_v2(nO=N_OUT, nI=N_HIDDEN)
    model = chain(linear, softmax)
    sample_X = rng.standard_normal((2, N_IN)).astype(np.float32)
    model.initialize(X=sample_X)

    # Save the raw bytes for Go to load.
    bytes_path = GOLDEN / "tiny_thinc_model.msgpack"
    bytes_path.write_bytes(model.to_bytes())
    print(f"wrote {bytes_path} ({bytes_path.stat().st_size} bytes)")

    # Build a fixed input for testing.
    X = rng.standard_normal((3, N_IN)).astype(np.float32)

    # Run forward pass through each sub-model individually so we can
    # capture every intermediate tensor.
    linear_out, _ = linear(X, is_train=False)
    softmax_out, _ = softmax(linear_out, is_train=False)

    # Final whole-model output (should match softmax_out).
    final_out = model.predict(X)

    assert linear_out.shape == (3, N_HIDDEN), linear_out.shape
    assert softmax_out.shape == (3, N_OUT), softmax_out.shape
    np.testing.assert_allclose(final_out, softmax_out, atol=1e-6)

    io_payload = {
        "thinc_version": thinc.__version__,
        "model_description": "chain(Linear(3, 4), Softmax_v2(4, 2))",
        "dims": {"nI": N_IN, "nHidden": N_HIDDEN, "nO": N_OUT},
        "input": array_to_json(X),
        "layer_outputs": {
            "linear": array_to_json(linear_out),
            "softmax": array_to_json(softmax_out),
        },
        "final_output": array_to_json(final_out),
    }
    io_path = GOLDEN / "tiny_thinc_model_io.json"
    io_path.write_text(json.dumps(io_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {io_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
