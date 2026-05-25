package gonum

import "math"

// Softmax applies row-wise softmax to a (n, k) row-major matrix X.
//
// Uses the standard numerically-stable form:
//
//	softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)) for j in row)
//
// out must be pre-allocated with len(out) == n*k. X and out may alias.
func Softmax(out []float32, X []float32, n, k int) {
	for i := 0; i < n; i++ {
		row := X[i*k : (i+1)*k]
		outRow := out[i*k : (i+1)*k]

		m := row[0]
		for j := 1; j < k; j++ {
			if row[j] > m {
				m = row[j]
			}
		}

		var sum float64
		for j := 0; j < k; j++ {
			e := math.Exp(float64(row[j] - m))
			outRow[j] = float32(e)
			sum += e
		}

		invSum := float32(1.0 / sum)
		for j := 0; j < k; j++ {
			outRow[j] *= invSum
		}
	}
}
