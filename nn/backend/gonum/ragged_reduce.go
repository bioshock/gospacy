package gonum

// ReduceFirst takes the first row of each sequence in a ragged batch.
//
// X is (T, w) row-major, concatenating B sequences.
// lengths is (B,) int32 with sum == T.
// out is (B, w) row-major; out[i] = X[startOf(i)].
func ReduceFirst(out []float32, X []float32, T, w int, lengths []int32) {
	off := 0
	for i, length := range lengths {
		copy(out[i*w:(i+1)*w], X[off*w:(off+1)*w])
		off += int(length)
	}
}

// ReduceLast takes the last row of each sequence in a ragged batch.
func ReduceLast(out []float32, X []float32, T, w int, lengths []int32) {
	off := 0
	for i, length := range lengths {
		last := off + int(length) - 1
		copy(out[i*w:(i+1)*w], X[last*w:(last+1)*w])
		off += int(length)
	}
}
