package gonum

// GatherAdd computes per-row sums of embedding lookups:
//
//	out[i, :] = sum_k table[indices[i, k], :]
//
// table:   (T, w) row-major
// indices: (N, K) row-major int32
// out:     (N, w) row-major; pre-allocated, fully written
func GatherAdd(out []float32, table []float32, T, w int, indices []int32, N, K int) {
	for i := 0; i < N; i++ {
		row := out[i*w : (i+1)*w]
		for j := range row {
			row[j] = 0
		}
		for k := 0; k < K; k++ {
			idx := int(indices[i*K+k])
			src := table[idx*w : (idx+1)*w]
			for j := 0; j < w; j++ {
				row[j] += src[j]
			}
		}
	}
}
