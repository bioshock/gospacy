"""Dump the StaticVectors W matrix from a populated bundle (md/lg) plus a
small set of golden lookups for use in nn/layers parity tests.

Reads the active model (GOSPACY_MODEL env var, default en_core_web_sm — which
has no StaticVectors and exits with a clear error). Writes the result to
testdata/golden/static_vectors{suffix}.json with:

  - "nO": projection output width (96 for md/lg)
  - "nM": vector-table column dim (300 for md/lg)
  - "W":  the StaticVectors layer's `W` param flattened row-major (length nO*nM)
  - "samples": [{ "key": str, "row": int, "out": [float32]*nO }, ...]

The sample words are deterministic English tokens with high coverage probability
in md/lg's vector table. Each `out` is computed via spacy's exact gemm path so
the Go side can assert byte-for-byte parity (well, within float32 1e-6).
"""

import json
import sys
from pathlib import Path

import numpy as np
import spacy

# Inline import so this script doesn't depend on common.py's load_nlp (which
# would error if MODEL_PATH points at sm). Replicate the env-var logic locally.
import os

REPO = Path(__file__).resolve().parent.parent
MODEL_NAME = os.environ.get("GOSPACY_MODEL", "en_core_web_sm")
MODEL_PATH = REPO / "testdata" / "models" / MODEL_NAME
GOLDEN = REPO / "testdata" / "golden"


def golden_suffix() -> str:
    if MODEL_NAME == "en_core_web_sm":
        return ""
    return "_" + MODEL_NAME.split("_")[-1]


def find_static_vectors_node(model):
    for node in model.walk():
        if node.name == "static_vectors":
            return node
    return None


def main():
    if MODEL_NAME == "en_core_web_sm":
        print("en_core_web_sm has no StaticVectors arm; skip", file=sys.stderr)
        sys.exit(0)
    if not MODEL_PATH.exists():
        print(f"missing model: {MODEL_PATH}; run `make download-assets`", file=sys.stderr)
        sys.exit(1)
    nlp = spacy.load(str(MODEL_PATH))
    sv = find_static_vectors_node(nlp.get_pipe("tok2vec").model)
    if sv is None:
        print(f"no static_vectors node in {MODEL_NAME}", file=sys.stderr)
        sys.exit(1)
    nO = sv.get_dim("nO")
    nM = sv.get_dim("nM")
    W = sv.get_param("W")  # (nO, nM)
    assert W.shape == (nO, nM), f"unexpected W shape {W.shape}"

    vocab = nlp.vocab
    vectors = vocab.vectors
    # Sample words: high-coverage English tokens for parity testing. Include
    # a clearly-OOV item to verify the -1 → zero-row contract.
    sample_words = [
        "apple", "the", "Google", "company", "running", "United",
        "Kingdom", "billion", "buying", "looking", "asdfghjklqwerty1234",
    ]
    samples = []
    for w in sample_words:
        key = vocab.strings.add(w)
        rows = vectors.find(keys=np.array([key], dtype=np.uint64))
        row = int(rows[0])
        if row < 0:
            out = np.zeros(nO, dtype=np.float32)
        else:
            V = vectors.data[row : row + 1]  # (1, nM)
            out = (V @ W.T).flatten().astype(np.float32)
        samples.append(
            {
                "key": w,
                "key_hash": int(key),
                "row": row,
                "out": out.tolist(),
            }
        )

    # Per-sample golden — includes the resolved row vector AND the expected
    # gemm output, so the Go parity test can pin both the lookup step and the
    # projection step independently.
    for s in samples:
        if s["row"] >= 0:
            s["vec"] = vectors.data[s["row"]].astype(np.float32).tolist()
        else:
            s["vec"] = None

    payload = {
        "nO": int(nO),
        "nM": int(nM),
        # W is written as a separate file (static_vectors_W{suffix}.bin) so
        # the JSON stays small. The bin file is raw little-endian float32
        # row-major (nO*nM*4 bytes).
        "W_file": f"static_vectors_W{golden_suffix()}.bin",
        "samples": samples,
    }

    GOLDEN.mkdir(parents=True, exist_ok=True)
    out_path = GOLDEN / f"static_vectors{golden_suffix()}.json"
    out_path.write_text(json.dumps(payload, indent=None))
    w_path = GOLDEN / payload["W_file"]
    W.astype("<f4").tofile(w_path)
    print(f"wrote {out_path}  (nO={nO}, nM={nM}, samples={len(samples)})")
    print(f"wrote {w_path}  ({W.nbytes} bytes)")


if __name__ == "__main__":
    main()
