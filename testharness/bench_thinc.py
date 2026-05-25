#!/usr/bin/env python3
"""Time thinc.NumpyOps for the same workloads as nn/backend/gonum/bench_test.go.

Output is one line per benchmark:
  <name>  <iterations>  <ns/op>

Run with:
  testharness/.venv/bin/python testharness/bench_thinc.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from thinc.api import NumpyOps

DEFAULT_REPS = 5_000


def time_op(name: str, fn, reps: int = DEFAULT_REPS):
    for _ in range(max(10, reps // 100)):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter_ns()
    ns_per_op = (t1 - t0) / reps
    print(f"{name:<40s}  {reps:>10d}  {ns_per_op:>15.1f} ns/op")


def main() -> int:
    ops = NumpyOps()
    rng = np.random.default_rng(seed=42)

    A = rng.standard_normal((30, 96)).astype(np.float32)
    B = rng.standard_normal((96, 300)).astype(np.float32)
    time_op("Gemm_30x96_96x300", lambda: ops.gemm(A, B))

    Wa = rng.standard_normal((300, 96)).astype(np.float32)
    ba = rng.standard_normal((300,)).astype(np.float32)
    time_op("Affine_30x96_300x96", lambda: ops.affine(A, Wa, ba))

    X_s = rng.standard_normal((30, 96)).astype(np.float32)
    time_op("Seq2Col_30x96_nW1", lambda: ops.seq2col(X_s, 1))

    X_m = rng.standard_normal((30, 96, 3)).astype(np.float32)
    time_op("Maxout_30x96_p3", lambda: ops.maxout(X_m))

    X_mish = rng.standard_normal((30, 96)).astype(np.float32)
    time_op("Mish_30x96", lambda: ops.mish(X_mish))

    X_sm = rng.standard_normal((30, 50)).astype(np.float32)
    time_op("Softmax_30x50", lambda: ops.softmax(X_sm))

    ids = np.arange(30, dtype=np.uint64) * 31337
    time_op("Hash_30ids", lambda: ops.hash(ids, 0))

    table = rng.standard_normal((1000, 96)).astype(np.float32)
    indices = (np.arange(30 * 4) % 1000).astype("i").reshape(30, 4)
    time_op("GatherAdd_30tokens_4lookups_96dim", lambda: ops.gather_add(table, indices))

    return 0


if __name__ == "__main__":
    sys.exit(main())
