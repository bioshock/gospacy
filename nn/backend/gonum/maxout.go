package gonum

// Maxout reduces the innermost dimension by taking the max across p pieces.
//
// Input X is logically (n, h, p) row-major:
//
//	X[i*h*p + j*p + k] = element at (i, j, k)
//
// Output:
//
//	out[i*h + j]   = max over k of X[i*h*p + j*p + k]
//	which[i*h + j] = argmax over k (used for backprop; not used at inference)
//
// All slices must be pre-allocated. len(out) == len(which) == n*h, len(X) == n*h*p.
func Maxout(out []float32, which []int32, X []float32, n, h, p int) {
	for i := 0; i < n; i++ {
		for j := 0; j < h; j++ {
			base := i*h*p + j*p
			maxVal := X[base]
			maxIdx := int32(0)
			for k := 1; k < p; k++ {
				v := X[base+k]
				if v > maxVal {
					maxVal = v
					maxIdx = int32(k)
				}
			}
			out[i*h+j] = maxVal
			which[i*h+j] = maxIdx
		}
	}
}
