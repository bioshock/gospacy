// Package nn provides the abstract Ops interface used by neural-network
// layers. Concrete implementations live in nn/backend/gonum (pure-Go default)
// and nn/backend/blis (cgo+BLIS, opt-in via -tags blis; not implemented in Phase 1a).
package nn

// Ops is the surface every backend implements. All output slices are
// pre-allocated by the caller; backends never allocate inside these methods
// (no GC pressure in hot paths).
type Ops interface {
	// Gemm: out = A @ B. A is m×k, B is k×n, out is m×n; all row-major float32.
	Gemm(out []float32, A []float32, m, k int, B []float32, n int)
	// Affine: out = X @ W.T + b. X is m×k, W is n×k (stored transposed), b is n-vector, out is m×n.
	// Matches thinc's convention where W has shape (nO, nI).
	Affine(out []float32, X []float32, m, k int, W []float32, n int, b []float32)
	// Seq2Col: expand (n, w) into (n, (2*nW+1)*w) windowed view, zero-padded.
	Seq2Col(out []float32, X []float32, n, w, nW int)
	// Maxout: piecewise max across the innermost dim. X is (n, h, p),
	// out is (n, h), which is (n, h) int32 argmax.
	Maxout(out []float32, which []int32, X []float32, n, h, p int)
	// Mish: out = X * tanh(softplus(X)), element-wise. threshold short-circuits large x.
	Mish(out []float32, X []float32, threshold float32)
	// Softmax: row-wise on (n, k). Numerically stable.
	Softmax(out []float32, X []float32, n, k int)
	// Hash: 4 hashes per uint64 id via thinc's custom 64-bit routine. out is N*4 uint32.
	Hash(out []uint32, ids []uint64, seed uint32)
	// GatherAdd: sum embedding-table lookups per token. table (T,w), indices (N,K), out (N,w).
	GatherAdd(out []float32, table []float32, T, w int, indices []int32, N, K int)
	// ReduceFirst / ReduceLast: first/last row of each ragged sequence.
	ReduceFirst(out []float32, X []float32, T, w int, lengths []int32)
	ReduceLast(out []float32, X []float32, T, w int, lengths []int32)
	// Pad: ragged → batch-major (B, max_len, w). Zero-padded.
	Pad(out []float32, X []float32, T, w int, lengths []int32, maxLen int)
	// List2Padded: Ragged → time-major Padded with bookkeeping arrays.
	List2Padded(data []float32, sizeAtT, sortedLengths, indices []int32,
		X []float32, T, w int, lengths []int32, maxLen int)
	// Padded2List: time-major Padded → Ragged in original order.
	Padded2List(out, paddedData []float32, sizeAtT, sortedLengths, indices []int32,
		B, T, w int, outLengths []int32)
}
