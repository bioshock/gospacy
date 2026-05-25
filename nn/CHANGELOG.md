# nn changelog

This package's changelog is consolidated in [`../CHANGELOG.md`](../CHANGELOG.md)
from Phase 3 onward. The original Phase-2 entry is preserved below for
historical reference; new entries go to the top-level file.

---

## v0.0.1-alpha — 2026-05-18

(Original Phase-2 entry — see top-level CHANGELOG.md for newer entries.)

First alpha release of the gospacy `nn/` package — internal-consumer-only.

### Added

- `nn.Model` struct + `Walk()` (depth-first pre-order matching thinc).
- `nn.ForwardFunc` for layer-defined forward passes.
- Tensor types: `Floats2d`, `Floats3d`, `Ints1d`, `Ints2d`, `Uint64s1d`, `Ragged`, `Padded`, `FloatList`.
- `nn.Ops` interface with 13 methods.
- `nn/backend/gonum` — pure-Go default implementation of `Ops` using `gonum/blas/blas32`.
- `nn/backend/blis` — cgo+BLIS scaffold behind `-tags blis`.
- `nn/layers` package — 6 layers and 7 combinators.
- `(*Model).FromBytes(b []byte) error`.
