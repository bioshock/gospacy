package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// Affine computes out = X @ W.T + b, where b is broadcast across rows.
// X is m×k, W is n×k (stored transposed relative to the weight matrix),
// b is length n, out is m×n.
//
// This matches thinc's affine convention: Y = X @ W.T + b.
func Affine(out []float32, X []float32, m, k int, W []float32, n int, b []float32) {
	blas32.Gemm(blas.NoTrans, blas.Trans,
		1.0,
		blas32.General{Rows: m, Cols: k, Stride: k, Data: X},
		blas32.General{Rows: n, Cols: k, Stride: k, Data: W},
		0.0,
		blas32.General{Rows: m, Cols: n, Stride: n, Data: out},
	)
	// Broadcast-add bias across rows
	for i := 0; i < m; i++ {
		row := out[i*n : (i+1)*n]
		for j := 0; j < n; j++ {
			row[j] += b[j]
		}
	}
}
