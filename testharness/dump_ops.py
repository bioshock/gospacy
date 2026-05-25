#!/usr/bin/env python3
"""Dump thinc.NumpyOps reference outputs for each op tested in Phase 1a.

Each op has its own output file at testdata/golden/ops-<op>.json with this schema:

  {
    "op": "gemm",
    "thinc_version": "9.x.x",
    "seed": 42,
    "cases": [
      {
        "name": "basic_3x4_4x5",
        "inputs": {
          "A": {"shape": [3,4], "dtype": "float32", "data": [...]},
          "B": {"shape": [4,5], "dtype": "float32", "data": [...]}
        },
        "output": {"shape":[3,5], "dtype":"float32", "data":[...]}
      },
      ...
    ]
  }

Per-op dump functions are added incrementally as each op task lands.
Run `python testharness/dump_ops.py <op_name|all>` to regenerate specific files.
"""

import json
import sys
from pathlib import Path

import numpy as np
import thinc
from thinc.api import NumpyOps

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"
SEED = 42


def make_rng() -> np.random.Generator:
    return np.random.default_rng(seed=SEED)


def array_to_json(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.flatten().tolist(),
    }


def write_op(op_name: str, cases: list) -> None:
    payload = {
        "op": op_name,
        "thinc_version": thinc.__version__,
        "seed": SEED,
        "cases": cases,
    }
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / f"ops-{op_name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


# --- Per-op dumpers (added incrementally) ---

def dump_gemm(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    # Case 1: basic 3x4 @ 4x5
    A = rng.standard_normal((3, 4)).astype(np.float32)
    B = rng.standard_normal((4, 5)).astype(np.float32)
    out = ops.gemm(A, B)
    cases.append({
        "name": "basic_3x4_4x5",
        "inputs": {"A": array_to_json(A), "B": array_to_json(B), "m": 3, "k": 4, "n": 5},
        "output": array_to_json(out),
    })
    # Case 2: 1x1 (degenerate)
    A = np.array([[2.0]], dtype=np.float32)
    B = np.array([[3.0]], dtype=np.float32)
    out = ops.gemm(A, B)
    cases.append({
        "name": "scalar_1x1",
        "inputs": {"A": array_to_json(A), "B": array_to_json(B), "m": 1, "k": 1, "n": 1},
        "output": array_to_json(out),
    })
    # Case 3: zeros (sanity)
    A = np.zeros((2, 3), dtype=np.float32)
    B = rng.standard_normal((3, 4)).astype(np.float32)
    out = ops.gemm(A, B)
    cases.append({
        "name": "zeros_2x3_3x4",
        "inputs": {"A": array_to_json(A), "B": array_to_json(B), "m": 2, "k": 3, "n": 4},
        "output": array_to_json(out),
    })
    return cases


def dump_affine(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    # thinc affine: Y = X @ W.T + b  → W shape is (n, k)
    # Case 1: X is 3x4, W is 5x4 (so W.T is 4x5), b is 5-vector → out is 3x5
    X = rng.standard_normal((3, 4)).astype(np.float32)
    W = rng.standard_normal((5, 4)).astype(np.float32)
    b = rng.standard_normal((5,)).astype(np.float32)
    out = ops.affine(X, W, b)
    cases.append({
        "name": "basic_3x4_4x5_b5",
        "inputs": {"X": array_to_json(X), "W": array_to_json(W), "b": array_to_json(b),
                   "m": 3, "k": 4, "n": 5},
        "output": array_to_json(out),
    })
    # Case 2: 1-row input; X is 1x3, W is 2x3 (W.T is 3x2), b is 2-vector → out is 1x2
    X = rng.standard_normal((1, 3)).astype(np.float32)
    W = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((2,)).astype(np.float32)
    out = ops.affine(X, W, b)
    cases.append({
        "name": "single_row_1x3_3x2_b2",
        "inputs": {"X": array_to_json(X), "W": array_to_json(W), "b": array_to_json(b),
                   "m": 1, "k": 3, "n": 2},
        "output": array_to_json(out),
    })
    return cases


def dump_seq2col(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    # Case 1: n=4 sequences, w=3 features, window=1
    X = rng.standard_normal((4, 3)).astype(np.float32)
    out = ops.seq2col(X, 1)
    cases.append({
        "name": "n4_w3_nW1",
        "inputs": {"X": array_to_json(X), "n": 4, "w": 3, "nW": 1},
        "output": array_to_json(out),
    })
    # Case 2: window=2
    X = rng.standard_normal((5, 4)).astype(np.float32)
    out = ops.seq2col(X, 2)
    cases.append({
        "name": "n5_w4_nW2",
        "inputs": {"X": array_to_json(X), "n": 5, "w": 4, "nW": 2},
        "output": array_to_json(out),
    })
    # Case 3: single token (edges fully padded)
    X = rng.standard_normal((1, 3)).astype(np.float32)
    out = ops.seq2col(X, 1)
    cases.append({
        "name": "single_token",
        "inputs": {"X": array_to_json(X), "n": 1, "w": 3, "nW": 1},
        "output": array_to_json(out),
    })
    return cases


def dump_maxout(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    X = rng.standard_normal((3, 5, 2)).astype(np.float32)  # n=3, h=5, p=2
    out, which = ops.maxout(X)
    cases.append({
        "name": "n3_h5_p2",
        "inputs": {"X": array_to_json(X), "n": 3, "h": 5, "p": 2},
        "output": array_to_json(out),
        "extra": {"which": {"shape": list(which.shape), "dtype": "int32",
                            "data": [int(x) for x in which.flatten()]}},
    })
    X = rng.standard_normal((2, 3, 4)).astype(np.float32)
    out, which = ops.maxout(X)
    cases.append({
        "name": "n2_h3_p4",
        "inputs": {"X": array_to_json(X), "n": 2, "h": 3, "p": 4},
        "output": array_to_json(out),
        "extra": {"which": {"shape": list(which.shape), "dtype": "int32",
                            "data": [int(x) for x in which.flatten()]}},
    })
    return cases


def dump_mish(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    X = rng.standard_normal((10,)).astype(np.float32) * 2
    out = ops.mish(X)
    cases.append({
        "name": "rand_10",
        "inputs": {"X": array_to_json(X)},
        "output": array_to_json(out),
    })
    X = np.array([-30.0, -1.0, 0.0, 1.0, 30.0, 50.0, -50.0], dtype=np.float32)
    out = ops.mish(X)
    cases.append({
        "name": "extreme_values",
        "inputs": {"X": array_to_json(X)},
        "output": array_to_json(out),
    })
    X = rng.standard_normal((3, 4)).astype(np.float32)
    out = ops.mish(X)
    cases.append({
        "name": "2d_3x4",
        "inputs": {"X": array_to_json(X)},
        "output": array_to_json(out),
    })
    return cases


def dump_softmax(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    X = rng.standard_normal((3, 5)).astype(np.float32)
    out = ops.softmax(X)
    cases.append({
        "name": "rowwise_3x5",
        "inputs": {"X": array_to_json(X), "n": 3, "k": 5},
        "output": array_to_json(out),
    })
    X = np.array([[100.0, 100.1, 99.9], [-100.0, -100.1, -99.9]], dtype=np.float32)
    out = ops.softmax(X)
    cases.append({
        "name": "extreme_values",
        "inputs": {"X": array_to_json(X), "n": 2, "k": 3},
        "output": array_to_json(out),
    })
    return cases


def dump_hash(ops: NumpyOps) -> list:
    cases = []
    ids = np.array([0, 1, 42, 100, 12345, 0xDEADBEEF, 2**63], dtype=np.uint64)
    out = ops.hash(ids, 0)
    cases.append({
        "name": "mixed_ids_seed0",
        "inputs": {"ids": {"shape": list(ids.shape), "dtype": "uint64",
                           "data": [int(x) for x in ids]},
                   "seed": 0, "n": int(len(ids))},
        "output": {"shape": list(out.shape), "dtype": "uint32",
                   "data": [int(x) for x in out.flatten()]},
    })
    out = ops.hash(ids, 7)
    cases.append({
        "name": "mixed_ids_seed7",
        "inputs": {"ids": {"shape": list(ids.shape), "dtype": "uint64",
                           "data": [int(x) for x in ids]},
                   "seed": 7, "n": int(len(ids))},
        "output": {"shape": list(out.shape), "dtype": "uint32",
                   "data": [int(x) for x in out.flatten()]},
    })
    return cases


def dump_gather_add(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    table = rng.standard_normal((10, 4)).astype(np.float32)
    indices = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    out = ops.gather_add(table, indices)
    cases.append({
        "name": "table10x4_idx3x2",
        "inputs": {"table": array_to_json(table),
                   "indices": {"shape": [3, 2], "dtype": "int32",
                               "data": [int(x) for x in indices.flatten()]},
                   "T": 10, "w": 4, "N": 3, "K": 2},
        "output": array_to_json(out),
    })
    return cases


def dump_reduce_first(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    X = rng.standard_normal((7, 4)).astype(np.float32)
    lengths = np.array([2, 3, 2], dtype=np.int32)
    out, _ = ops.reduce_first(X, lengths)
    cases.append({
        "name": "3seqs_total7_w4",
        "inputs": {"X": array_to_json(X),
                   "lengths": {"shape": [3], "dtype": "int32", "data": [2, 3, 2]},
                   "T": 7, "w": 4, "B": 3},
        "output": array_to_json(out),
    })
    return cases


def dump_reduce_last(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    X = rng.standard_normal((7, 4)).astype(np.float32)
    lengths = np.array([2, 3, 2], dtype=np.int32)
    out, _ = ops.reduce_last(X, lengths)
    cases.append({
        "name": "3seqs_total7_w4",
        "inputs": {"X": array_to_json(X),
                   "lengths": {"shape": [3], "dtype": "int32", "data": [2, 3, 2]},
                   "T": 7, "w": 4, "B": 3},
        "output": array_to_json(out),
    })
    return cases


def dump_list2padded(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    seqs = [
        rng.standard_normal((2, 3)).astype(np.float32),
        rng.standard_normal((4, 3)).astype(np.float32),
        rng.standard_normal((1, 3)).astype(np.float32),
    ]
    padded = ops.list2padded(seqs)
    X_concat = np.concatenate(seqs, axis=0)
    cases.append({
        "name": "3seqs_w3",
        "inputs": {
            "X": array_to_json(X_concat),
            "lengths": {"shape": [3], "dtype": "int32", "data": [2, 4, 1]},
            "T": 7, "w": 3, "B": 3, "max_len": 4,
        },
        "output": array_to_json(np.asarray(padded.data)),
        "extra": {
            "size_at_t": {"shape": list(padded.size_at_t.shape), "dtype": "int32",
                          "data": [int(x) for x in padded.size_at_t]},
            "indices":   {"shape": list(padded.indices.shape), "dtype": "int32",
                          "data": [int(x) for x in padded.indices]},
            "sorted_lengths": {"shape": list(padded.lengths.shape), "dtype": "int32",
                               "data": [int(x) for x in padded.lengths]},
        },
    })
    return cases


def dump_padded2list(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    seqs_in = [
        rng.standard_normal((2, 3)).astype(np.float32),
        rng.standard_normal((4, 3)).astype(np.float32),
        rng.standard_normal((1, 3)).astype(np.float32),
    ]
    padded = ops.list2padded(seqs_in)
    seqs_out = ops.padded2list(padded)
    out_concat = np.concatenate(seqs_out, axis=0)
    cases.append({
        "name": "roundtrip_3seqs",
        "inputs": {
            "padded_data": array_to_json(np.asarray(padded.data)),
            "size_at_t":   {"shape": list(padded.size_at_t.shape), "dtype": "int32",
                            "data": [int(x) for x in padded.size_at_t]},
            "sorted_lengths": {"shape": list(padded.lengths.shape), "dtype": "int32",
                               "data": [int(x) for x in padded.lengths]},
            "indices":     {"shape": list(padded.indices.shape), "dtype": "int32",
                            "data": [int(x) for x in padded.indices]},
            "B": 3, "T": 4, "w": 3,
            "out_lengths": {"shape": [3], "dtype": "int32", "data": [2, 4, 1]},
        },
        "output": array_to_json(out_concat),
    })
    return cases


def dump_pad(ops: NumpyOps) -> list:
    rng = make_rng()
    cases = []
    seqs = [
        rng.standard_normal((2, 3)).astype(np.float32),
        rng.standard_normal((4, 3)).astype(np.float32),
        rng.standard_normal((1, 3)).astype(np.float32),
    ]
    X_concat = np.concatenate(seqs, axis=0)
    padded = ops.pad(seqs)
    cases.append({
        "name": "3seqs_w3",
        "inputs": {"X": array_to_json(X_concat),
                   "lengths": {"shape": [3], "dtype": "int32", "data": [2, 4, 1]},
                   "T": 7, "w": 3, "B": 3, "max_len": 4},
        "output": array_to_json(np.asarray(padded)),
    })
    return cases


# Map op-name → dumper function. Extended in each subsequent op task.
DUMPERS = {
    "gemm": dump_gemm,
    "affine": dump_affine,
    "seq2col": dump_seq2col,
    "maxout": dump_maxout,
    "mish": dump_mish,
    "softmax": dump_softmax,
    "hash": dump_hash,
    "gather_add": dump_gather_add,
    "reduce_first": dump_reduce_first,
    "reduce_last": dump_reduce_last,
    "pad": dump_pad,
    "list2padded": dump_list2padded,
    "padded2list": dump_padded2list,
}


def dump_sample_ops(ops: NumpyOps) -> None:
    """Tiny per-op fixtures used by Go unit tests without `make diff-test`."""
    payload = {"thinc_version": thinc.__version__, "ops": {}}
    rng = make_rng()
    # Just one tiny case per op available so far.
    if "gemm" in DUMPERS:
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        out = ops.gemm(A, B)
        payload["ops"]["gemm"] = {
            "name": "tiny_2x2_2x2",
            "inputs": {"A": array_to_json(A), "B": array_to_json(B), "m": 2, "k": 2, "n": 2},
            "output": array_to_json(out),
        }
    if "affine" in DUMPERS:
        # thinc affine: Y = X @ W.T + b → W is (n, k)
        # X is 1x2, W is 2x2 identity (W.T = identity), b = [10, 20] → out = [11, 22]
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        W = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)
        out = ops.affine(X, W, b)
        payload["ops"]["affine"] = {
            "name": "tiny_identity_plus_b",
            "inputs": {"X": array_to_json(X), "W": array_to_json(W), "b": array_to_json(b),
                       "m": 1, "k": 2, "n": 2},
            "output": array_to_json(out),
        }
    if "seq2col" in DUMPERS:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        out = ops.seq2col(X, 1)
        payload["ops"]["seq2col"] = {
            "name": "tiny_3x2_nW1",
            "inputs": {"X": array_to_json(X), "n": 3, "w": 2, "nW": 1},
            "output": array_to_json(out),
        }
    if "maxout" in DUMPERS:
        X = np.array([[[1.0, 2.0], [5.0, 3.0]]], dtype=np.float32)  # n=1, h=2, p=2
        out, which = ops.maxout(X)
        payload["ops"]["maxout"] = {
            "name": "tiny_1x2x2",
            "inputs": {"X": array_to_json(X), "n": 1, "h": 2, "p": 2},
            "output": array_to_json(out),
            "extra": {"which": {"shape": list(which.shape), "dtype": "int32",
                                "data": [int(x) for x in which.flatten()]}},
        }
    if "mish" in DUMPERS:
        X = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        out = ops.mish(X)
        payload["ops"]["mish"] = {
            "name": "tiny_4",
            "inputs": {"X": array_to_json(X)},
            "output": array_to_json(out),
        }
    if "softmax" in DUMPERS:
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = ops.softmax(X)
        payload["ops"]["softmax"] = {
            "name": "tiny_1x3",
            "inputs": {"X": array_to_json(X), "n": 1, "k": 3},
            "output": array_to_json(out),
        }
    if "hash" in DUMPERS:
        ids = np.array([42, 100], dtype=np.uint64)
        out = ops.hash(ids, 0)
        payload["ops"]["hash"] = {
            "name": "tiny_2_seed0",
            "inputs": {"ids": {"shape": [2], "dtype": "uint64", "data": [42, 100]},
                       "seed": 0, "n": 2},
            "output": {"shape": list(out.shape), "dtype": "uint32",
                       "data": [int(x) for x in out.flatten()]},
        }
    if "gather_add" in DUMPERS:
        table = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
        out = ops.gather_add(table, indices)
        payload["ops"]["gather_add"] = {
            "name": "tiny",
            "inputs": {"table": array_to_json(table),
                       "indices": {"shape": [2, 2], "dtype": "int32",
                                   "data": [0, 1, 1, 2]},
                       "T": 3, "w": 2, "N": 2, "K": 2},
            "output": array_to_json(out),
        }
    if "reduce_first" in DUMPERS:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        lengths = np.array([1, 2], dtype=np.int32)
        out, _ = ops.reduce_first(X, lengths)
        payload["ops"]["reduce_first"] = {
            "name": "tiny",
            "inputs": {"X": array_to_json(X),
                       "lengths": {"shape": [2], "dtype": "int32", "data": [1, 2]},
                       "T": 3, "w": 2, "B": 2},
            "output": array_to_json(out),
        }
    if "reduce_last" in DUMPERS:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        lengths = np.array([1, 2], dtype=np.int32)
        out, _ = ops.reduce_last(X, lengths)
        payload["ops"]["reduce_last"] = {
            "name": "tiny",
            "inputs": {"X": array_to_json(X),
                       "lengths": {"shape": [2], "dtype": "int32", "data": [1, 2]},
                       "T": 3, "w": 2, "B": 2},
            "output": array_to_json(out),
        }
    if "pad" in DUMPERS:
        seqs = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ]
        X_concat = np.concatenate(seqs, axis=0)
        padded = ops.pad(seqs)
        payload["ops"]["pad"] = {
            "name": "tiny_2seqs",
            "inputs": {"X": array_to_json(X_concat),
                       "lengths": {"shape": [2], "dtype": "int32", "data": [1, 2]},
                       "T": 3, "w": 2, "B": 2, "max_len": 2},
            "output": array_to_json(np.asarray(padded)),
        }
    if "list2padded" in DUMPERS:
        seqs = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ]
        padded = ops.list2padded(seqs)
        X_concat = np.concatenate(seqs, axis=0)
        payload["ops"]["list2padded"] = {
            "name": "tiny_2seqs",
            "inputs": {
                "X": array_to_json(X_concat),
                "lengths": {"shape": [2], "dtype": "int32", "data": [1, 2]},
                "T": 3, "w": 2, "B": 2, "max_len": 2,
            },
            "output": array_to_json(np.asarray(padded.data)),
            "extra": {
                "size_at_t": {"shape": list(padded.size_at_t.shape), "dtype": "int32",
                              "data": [int(x) for x in padded.size_at_t]},
                "indices":   {"shape": list(padded.indices.shape), "dtype": "int32",
                              "data": [int(x) for x in padded.indices]},
                "sorted_lengths": {"shape": list(padded.lengths.shape), "dtype": "int32",
                                   "data": [int(x) for x in padded.lengths]},
            },
        }
    if "padded2list" in DUMPERS:
        seqs_in = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ]
        padded = ops.list2padded(seqs_in)
        seqs_out = ops.padded2list(padded)
        out_concat = np.concatenate(seqs_out, axis=0)
        payload["ops"]["padded2list"] = {
            "name": "roundtrip_tiny",
            "inputs": {
                "padded_data": array_to_json(np.asarray(padded.data)),
                "size_at_t":   {"shape": list(padded.size_at_t.shape), "dtype": "int32",
                                "data": [int(x) for x in padded.size_at_t]},
                "sorted_lengths": {"shape": list(padded.lengths.shape), "dtype": "int32",
                                   "data": [int(x) for x in padded.lengths]},
                "indices":     {"shape": list(padded.indices.shape), "dtype": "int32",
                                "data": [int(x) for x in padded.indices]},
                "B": 2, "T": 2, "w": 2,
                "out_lengths": {"shape": [2], "dtype": "int32", "data": [1, 2]},
            },
            "output": array_to_json(out_concat),
        }
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / "sample_ops.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    ops = NumpyOps()
    if target == "all":
        for name, fn in DUMPERS.items():
            write_op(name, fn(ops))
        dump_sample_ops(ops)
    elif target == "sample":
        dump_sample_ops(ops)
    elif target in DUMPERS:
        write_op(target, DUMPERS[target](ops))
    else:
        print(f"unknown op: {target}", file=sys.stderr)
        print(f"available: {sorted(DUMPERS)} | all | sample", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
