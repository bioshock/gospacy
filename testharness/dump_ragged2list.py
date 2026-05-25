#!/usr/bin/env python3
"""Generate the ragged2list golden fixture."""

import json
from pathlib import Path

import numpy as np
from thinc.api import ragged2list
from thinc.types import Ragged

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"


def main() -> int:
    rng = np.random.default_rng(3)
    cols = 4
    lengths = np.array([3, 2, 5], dtype="i")
    total = int(lengths.sum())
    data = rng.standard_normal((total, cols)).astype(np.float32)

    model = ragged2list()
    Y, _ = model(Ragged(data, lengths), is_train=False)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "thinc ragged2list — Ragged(N=10, cols=4) → List[Floats2d]",
        "cols": cols,
        "lengths": lengths.tolist(),
        "input_data": data.flatten().tolist(),
        "output_items": [{"shape": list(item.shape), "data": item.flatten().tolist()} for item in Y],
    }
    (GOLDEN / "ragged2list.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {GOLDEN / 'ragged2list.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
