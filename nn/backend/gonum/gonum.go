// Package gonum implements the gospacy Ops interface using gonum/blas/blas32
// for BLAS-class ops and hand-written Go for non-BLAS ops. This is the default
// backend for `go get` users (no cgo).
package gonum

// Ops is the pure-Go implementation of nn.Ops. Construct with New().
// Stateless and safe for concurrent use.
type Ops struct{}

// New returns a pure-Go Ops implementation.
func New() *Ops { return &Ops{} }

// Gemm delegates to the package-level Gemm function.
func (*Ops) Gemm(out []float32, A []float32, m, k int, B []float32, n int) {
	Gemm(out, A, m, k, B, n)
}

// Affine delegates to the package-level Affine function.
func (*Ops) Affine(out []float32, X []float32, m, k int, W []float32, n int, b []float32) {
	Affine(out, X, m, k, W, n, b)
}

// Seq2Col delegates to the package-level Seq2Col function.
func (*Ops) Seq2Col(out []float32, X []float32, n, w, nW int) {
	Seq2Col(out, X, n, w, nW)
}

// Maxout delegates to the package-level Maxout function.
func (*Ops) Maxout(out []float32, which []int32, X []float32, n, h, p int) {
	Maxout(out, which, X, n, h, p)
}

// Mish delegates to the package-level Mish function.
func (*Ops) Mish(out []float32, X []float32, threshold float32) {
	Mish(out, X, threshold)
}

// Softmax delegates to the package-level Softmax function.
func (*Ops) Softmax(out []float32, X []float32, n, k int) {
	Softmax(out, X, n, k)
}

// Hash delegates to the package-level Hash function.
func (*Ops) Hash(out []uint32, ids []uint64, seed uint32) {
	Hash(out, ids, seed)
}

// GatherAdd delegates to the package-level GatherAdd function.
func (*Ops) GatherAdd(out []float32, table []float32, T, w int, indices []int32, N, K int) {
	GatherAdd(out, table, T, w, indices, N, K)
}

// ReduceFirst delegates to the package-level ReduceFirst function.
func (*Ops) ReduceFirst(out []float32, X []float32, T, w int, lengths []int32) {
	ReduceFirst(out, X, T, w, lengths)
}

// ReduceLast delegates to the package-level ReduceLast function.
func (*Ops) ReduceLast(out []float32, X []float32, T, w int, lengths []int32) {
	ReduceLast(out, X, T, w, lengths)
}

// Pad delegates to the package-level Pad function.
func (*Ops) Pad(out []float32, X []float32, T, w int, lengths []int32, maxLen int) {
	Pad(out, X, T, w, lengths, maxLen)
}

// List2Padded delegates to the package-level List2Padded function.
func (*Ops) List2Padded(data []float32, sizeAtT, sortedLengths, indices []int32,
	X []float32, T, w int, lengths []int32, maxLen int) {
	List2Padded(data, sizeAtT, sortedLengths, indices, X, T, w, lengths, maxLen)
}

// Padded2List delegates to the package-level Padded2List function.
func (*Ops) Padded2List(out, paddedData []float32, sizeAtT, sortedLengths, indices []int32,
	B, T, w int, outLengths []int32) {
	Padded2List(out, paddedData, sizeAtT, sortedLengths, indices, B, T, w, outLengths)
}
