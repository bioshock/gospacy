package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// Gemm computes out = A @ B for row-major float32 matrices.
// A is m×k, B is k×n, out is m×n. All slices must be pre-allocated to the
// correct length; out is fully written (existing values are overwritten).
//
// This is the no-transpose variant. Transposed gemm is added when a layer needs it.
func Gemm(out []float32, A []float32, m, k int, B []float32, n int) {
	blas32.Gemm(blas.NoTrans, blas.NoTrans,
		1.0,
		blas32.General{Rows: m, Cols: k, Stride: k, Data: A},
		blas32.General{Rows: k, Cols: n, Stride: n, Data: B},
		0.0,
		blas32.General{Rows: m, Cols: n, Stride: n, Data: out},
	)
}
