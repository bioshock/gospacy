package gonum

import "math"

// Mish computes the Mish activation element-wise: out[i] = X[i] * tanh(softplus(X[i])).
//
// For x > threshold, mish(x) ≈ x; we use that identity to short-circuit.
// thinc uses threshold=20 by default. Uses float64 internally for stability,
// then casts back to float32. out and X may alias.
func Mish(out []float32, X []float32, threshold float32) {
	for i, x := range X {
		if x > threshold {
			out[i] = x
			continue
		}
		x64 := float64(x)
		sp := math.Log1p(math.Exp(x64))
		out[i] = float32(x64 * math.Tanh(sp))
	}
}
