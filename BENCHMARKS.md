# gospacy benchmarks

Per-op timings comparing the pure-Go (gonum) backend against Python `thinc.NumpyOps`. Absolute numbers will vary; ratios are the figure of merit.

## Methodology

- **Go**: `go test -bench=. -benchmem -run=^$ ./nn/...` against the `gonum.Ops` implementation. Reported as `ns/op` from `testing.B`.
- **Python**: `testharness/bench_thinc.py` calls `thinc.NumpyOps` methods 5000 times after warm-up. Reported as mean ns/op.
- Workloads roughly match a `en_core_web_sm` tagger inference: 30-token batch, 96-dim tok2vec width, 50-class tag head.
- Measured on:
  - **Machine**: `Linux system1 6.8.0-40-generic #40-Ubuntu SMP x86_64`
  - **Go**: `go1.26.1 linux/amd64`
  - **Python**: `thinc 3.8.14`
  - **NumPy BLAS**: `scipy-openblas / OpenBLAS 0.3.29 DYNAMIC_ARCH Haswell`

## Results

| Op | Go ns/op | Python ns/op | Go/Python ratio | Notes |
|---|---:|---:|---:|---|
| Gemm 30×96 @ 96×300 | 69,883 | 60,432 | 1.16× | BLAS-class via gonum/blas32 vs cython-blis |
| Affine 30×96 + bias | 73,974 | 65,094 | 1.14× | Gemm + bias add (Go) vs cython-blis (Python) |
| Seq2Col 30×96 nW=1 | 1,112 | 3,448 | **0.32×** | Hand-written Go loop beats thinc Cython |
| Maxout 30×96 p=3 | 8,541 | 5,481 | 1.56× | Hand-written Go vs thinc Cython |
| Mish 30×96 | 117,518 | 79,560 | 1.48× | float64 intermediates for exp/log/tanh |
| Softmax 30×50 | 21,753 | 11,784 | 1.85× | LogSumExp stable form |
| Hash 30 ids | 241 | 1,184 | **0.20×** | Custom 64-bit routine; Python has interpreter overhead |
| GatherAdd 30×4 lookups | 5,793 | 5,912 | 0.98× | Pure-Go iteration; essentially tied |
| Tiny Chain forward (Linear→Softmax) | 513 | n/a | n/a | Full forward pass; no Python equivalent benchmark |

## Interpretation

**All ops within 2× of Python.** No perf-pass action this phase.

**Notable wins**:
- **Seq2Col 3.1× faster than Python**: pure-Go cache-friendly loops avoid the Python→Cython call overhead at this batch size (30 tokens × 96 dim is small enough that op-call overhead dominates).
- **Hash 4.9× faster than Python**: Python's per-call wrap of a 30-element loop is dominated by interpreter overhead; Go inlines the per-id Murmur3 call.

**Acceptable gaps**:
- **Softmax 1.85×**: closest to the threshold. Cause: Go's `math.Exp` is float64-precision and slower than thinc's vectorised numpy. Optimising would require either a faster `expf` (libm via cgo) or a SIMD path — both out of scope for pure-Go.
- **Mish 1.48×**: same root cause — exp/log/tanh in libm. Acceptable.
- **Maxout 1.56×**: hand-written piece-wise max loop. Could be optimised by replacing `> maxVal` branch with a branchless update, but the 1.56× ratio is well within target.
- **Gemm/Affine 1.14-1.16×**: gonum/blas32 is competitive with cython-blis at this size. The 14-16% gap is the cost of pure-Go BLAS — totally acceptable.

**Allocations**: Gemm and Affine allocate 12 times per call (~1.2 KB total). This is inside gonum's blas32 implementation, not our wrapper. Non-trivial to remove without forking gonum. At 30×96 input scale, the allocator is not the bottleneck.

## Perf gaps and decisions

All hot kernels within 2× of Python. No perf-pass action this phase.

## Regenerating

```bash
make bootstrap-ref       # if .venv missing
go test -bench=. -benchmem -run=^$ ./nn/...
testharness/.venv/bin/python testharness/bench_thinc.py
```

Update this file with the new numbers after any code change to a hot op.

---

## Phase 7 Block B — Bundle.Pipe end-to-end timings (2026-05-21)

Block B revisits performance at `Bundle.Pipe` granularity (not per-op) to
diagnose a 2× Go-vs-Python latency gap surfaced on long claim-style input
(16.2 ms/record Python, 31.6 ms/record Go on a representative sample).

### Methodology

- **Go**: `go test -bench=^BenchmarkBundle_Pipe_LongClaimStyle$ -benchtime=10s -count=3 ./pipeline`,
  reported median of three runs. The 200-character text is a verbatim
  trademark-class description ("Computer software; application
  software; downloadable software for trading crypto-products …"),
  shaped like the comma-/semicolon-heavy claim text where the latency
  gap first showed up.
- **Python**: inline `cProfile` script under `testharness/.venv/bin/python`
  using `common.load_nlp()` on the same model + same text, 2000 iterations
  after 50-iteration warmup, mean ns/op.
- **Hardware**: AMD Ryzen 7 5700G (16 threads), Linux 6.8.0-40-generic
  x86_64.
- **Go**: go1.26.1 linux/amd64.
- **Python**: spaCy 3.8.14, en_core_web_sm.

### Results

| Workload | Go ns/op (before fix) | Go ns/op (after fix) | Python ns/op | Go(after) / Python |
|---|---:|---:|---:|---:|
| `BenchmarkBundle_Pipe_LongClaimStyle`   | 26,546,300 | 3,667,000 | 8,236,257 | **0.45×** |
| `BenchmarkBundle_Pipe_FixtureSentences` | 3,372,987 | 1,247,365 | n/a | n/a |

Allocations on the long-claim bench dropped from **23.4 MB/op, 18,559
allocs/op** to **1.37 MB/op, 6,154 allocs/op** — a 17× memory reduction
and 67% fewer allocations per record.

### Pre-fix hot path (Go)

From `go tool pprof -top -cum /tmp/gospacy_cpu.out` on the long-claim
bench:

```
      flat  flat%   sum%        cum   cum%
         0     0%     0%     12.70s 53.34%  pipeline.(*Lemmatizer).Apply
         0     0%     0%     12.70s 53.34%  pipeline.(*Lemmatizer).lemmatize
     0.87s  3.65%  3.65%     12.70s 53.34%  pipeline.(*Lemmatizer).ruleLemmatize
     0.88s  3.70%  7.35%     10.51s 44.14%  runtime.mapassign_faststr
     0.01s 0.042%  7.39%      7.17s 30.11%  runtime.systemstack
         0     0%  7.39%      6.63s 27.85%  internal/runtime/maps.(*table).rehash
     0.36s  1.51%  8.90%      6.56s 27.55%  internal/runtime/maps.(*table).split
     2.31s  9.70% 35.95%      2.31s  9.70%  aeshashbody
         0     0% 35.95%      2.22s  9.32%  gonum/blas/gonum.sgemmSerial
```

**Finding:** `Lemmatizer.ruleLemmatize` at 53% cumulative, with
`runtime.mapassign_faststr` alone at 44% cumulative. The cause:
`readExc(posHash)` and `readIndex(posHash)` each materialised a fresh
`map[string][]string` / `[]string` from the underlying lookup tables
**per token**, copying potentially thousands of entries just to read one
key. `ruleLemmatize` then built yet another `indexSet map[string]struct{}`
from that copy. For a 24-token long-claim sentence this paid the cost
~24× per `Pipe` call. Gemm only accounted for 9.3% — **gemm was not the
bottleneck**.

### Python hot path (cProfile, same input)

Top by cumulative time in `nlp(text)`:

```
ncalls   cumtime  function
  2000     19.26  language.py:1020(__call__)         # full pipeline
  8000     16.61  trainable_pipe.pyx:40(__call__)     # tagger+parser
  8000     11.98  thinc/model.py:330(predict)
 30000      5.86  thinc/backends/numpy_ops.pyx:91(gemm)
  4000      6.14  spacy/ml/tb_framework.py:31(forward) # parser tb
```

`gemm` is at 30% cumulative; layers (`maxout`, `chain`, `residual`,
`with_array`, `layernorm`) collectively account for the rest. **Python's
hot path is gemm.** The lemmatizer is not in Python's top-25 at all — its
implementation uses pre-loaded dicts and is O(1) per token.

### Optimisation applied (Phase 7 Block B Task B4)

`perf(pipeline): cache per-POS lemma payloads in Lemmatizer` (commit
1a82e68). A `posCache` struct on the Lemmatizer memoises
`{rules, excs, indexSet}` per POS hash, lazy-built on first sight and
reused for every subsequent token of that POS. `readExc / readIndex /
readRules` are kept and now run once per POS rather than once per token.
Behaviour preserved (real-bundle differentials, lemmatizer_test.go, all
17 packages remain green); only redundant rebuild work removed.

### Post-fix hot path (Go)

```
      flat  flat%   sum%        cum   cum%
         0     0%     0%     13.98s 53.06%  gonum/blas/gonum.sgemmSerial
     5.44s 20.65% 20.65%     13.98s 53.06%  gonum/blas/gonum.sgemmSerialNotTrans
     8.54s 32.41% 53.09%      8.54s 32.41%  gonum/internal/asm/f32.DotUnitary
         0     0% 53.13%      2.97s 11.27%  pipeline.(*Parser).Apply
         0     0% 53.24%      2.61s  9.91%  tokenizer.(*Tokenizer).Tokenize
```

Pipeline is now **BLAS-bound** (53% gemm) just like Python's profile.
`Lemmatizer.ruleLemmatize` is no longer in the top entries. The parser
inner loop and tokenizer regex matching account for most of the rest —
both pure-Go, both already lean.

### Decision (cgo+BLIS)

cgo+BLIS remains **deferred**. The original case (defer until an anchor
user surfaces a hard latency floor pure-Go cannot meet) now has stronger
evidence on both sides: gemm IS the dominant residual cost (53% cum), but
gospacy is **already ~2.2× faster than Python** on this workload
(3.67 ms vs 8.24 ms end-to-end). cgo+BLIS would widen that lead, not be
required for parity. The build-time `libblis-dev` requirement, cgo
barrier, and Windows reachability cost remain. See `NOT_YET_PORTED.md`'s
"Full cgo + BLIS bindings" section for the updated rationale.

### Speedup applied this phase

`perf(pipeline): cache per-POS lemma payloads in Lemmatizer`: **7.2×
speedup** on the long-claim-style bench (26.5 ms → 3.67 ms),
**2.7× speedup** on the fixture-sentences bench (3.37 ms → 1.25 ms),
**17× memory reduction** (23.4 MB → 1.37 MB per record).

### Reproducing

```bash
go test -run=^$ -bench=^BenchmarkBundle_Pipe -benchtime=10s -count=3 ./pipeline

testharness/.venv/bin/python -c "
import sys, time
sys.path.insert(0, 'testharness')
from common import load_nlp
nlp = load_nlp()
text = 'Computer software; application software; downloadable software for trading crypto-products and providing crypto-currency information; authentication and authorization software; automatic banking machines.'
for _ in range(50): nlp(text)
t0 = time.perf_counter()
for _ in range(2000): nlp(text)
print(int((time.perf_counter()-t0)*1e9/2000), 'ns/op')
"
```
