package gonum

// Seq2Col expands a (n, w) matrix into a (n, (2*nW+1)*w) matrix by concatenating
// each row with its nW neighbours on each side. Rows outside [0, n) are zero-padded.
//
// out must be pre-allocated with len(out) == n*(2*nW+1)*w.
// X must have len(X) == n*w.
//
// This is a hot path for CNN tok2vec; the loops are written for cache friendliness.
func Seq2Col(out []float32, X []float32, n, w, nW int) {
	for i := range out {
		out[i] = 0
	}
	outCols := (2*nW + 1) * w
	for i := 0; i < n; i++ {
		dstRow := out[i*outCols : (i+1)*outCols]
		for win := -nW; win <= nW; win++ {
			srcRow := i + win
			if srcRow < 0 || srcRow >= n {
				continue
			}
			dstOff := (win + nW) * w
			src := X[srcRow*w : (srcRow+1)*w]
			copy(dstRow[dstOff:dstOff+w], src)
		}
	}
}
