# Example: bench

Loads a `.spacy` bundle and times `Bundle.Pipe` over 100 short sentences. Prints
sentences/sec, tokens/sec, and microseconds per token.

This is a quick smoke-test of end-to-end throughput on your box. For per-op
microbenchmarks (gemm, seq2col, maxout, mish, hash, tok2vec forward), see
`nn/*_bench_test.go` and `BENCHMARKS.md` at the repo root.

## Running

One-time bundle download (from the repo root):

```bash
make bootstrap-ref
make download-assets
```

Then:

```bash
go run ./examples/bench ./testdata/models/en_core_web_sm
```

## Expected output

```
Sentences:  100
Tokens:     <total>
Elapsed:    <wall-clock>
Throughput: <sps> sentences/sec, <tps> tokens/sec
Per token:  <us> µs
```

Numbers are machine-dependent. On a 2026-era amd64 laptop, expect 5-50
sentences/sec in pure-Go mode (the parser dominates). The cgo+BLIS backend is
not yet exercised (see `NOT_YET_PORTED.md`).

## What this is NOT

- A regression-tracking benchmark — for that, see `BENCHMARKS.md` (which
  compares Go vs Python per-op timings).
- A profiling tool — use `go test -bench=. -cpuprofile=...` on individual
  packages for that.
