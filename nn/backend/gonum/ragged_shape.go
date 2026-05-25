package gonum

// List2Padded converts a concat'd ragged batch into a time-major padded tensor.
//
// thinc's Padded sorts sequences length-descending and tracks sizeAtT (count
// of sequences alive at each timestep) plus indices (map sorted-pos → original-pos).
//
// Inputs:
//
//	X:       concat'd ragged data, len = T*w.
//	T, w:    total rows and feature width.
//	lengths: per-sequence row count (in original order).
//	maxLen:  max(lengths).
//
// Outputs:
//
//	data:          (maxLen, B, w) time-major, zero-padded.
//	sizeAtT:       (maxLen,) count of sequences alive at each timestep.
//	sortedLengths: (B,) lengths after length-desc sort.
//	indices:       (B,) maps sorted-position i → original-position.
//
// All output slices must be pre-allocated.
func List2Padded(data []float32, sizeAtT, sortedLengths, indices []int32,
	X []float32, T, w int, lengths []int32, maxLen int) {

	B := len(lengths)

	for i := range indices {
		indices[i] = int32(i)
	}
	for i := 1; i < B; i++ {
		for j := i; j > 0 && lengths[indices[j-1]] < lengths[indices[j]]; j-- {
			indices[j-1], indices[j] = indices[j], indices[j-1]
		}
	}
	for i := 0; i < B; i++ {
		sortedLengths[i] = lengths[indices[i]]
	}

	starts := make([]int, B)
	off := 0
	for i, l := range lengths {
		starts[i] = off
		off += int(l)
	}

	for t := 0; t < maxLen; t++ {
		count := int32(0)
		for _, l := range lengths {
			if int(l) > t {
				count++
			}
		}
		sizeAtT[t] = count
	}

	for i := range data {
		data[i] = 0
	}
	for bSorted := 0; bSorted < B; bSorted++ {
		orig := int(indices[bSorted])
		L := int(lengths[orig])
		for t := 0; t < L; t++ {
			dst := (t*B + bSorted) * w
			src := (starts[orig] + t) * w
			copy(data[dst:dst+w], X[src:src+w])
		}
	}
}

// Padded2List reverses List2Padded: time-major padded → concat'd ragged in
// ORIGINAL order.
func Padded2List(out, paddedData []float32, sizeAtT, sortedLengths, indices []int32,
	B, T, w int, outLengths []int32) {

	starts := make([]int, B)
	off := 0
	for i, l := range outLengths {
		starts[i] = off
		off += int(l)
	}

	for bSorted := 0; bSorted < B; bSorted++ {
		orig := int(indices[bSorted])
		L := int(outLengths[orig])
		for t := 0; t < L; t++ {
			src := (t*B + bSorted) * w
			dst := (starts[orig] + t) * w
			copy(out[dst:dst+w], paddedData[src:src+w])
		}
	}
}

// Pad converts a ragged batch (concat'd (T, w) + lengths) into a padded
// (B, max_len, w) tensor. Rows beyond each sequence's length are zero.
//
// Memory layout: out is row-major (B, max_len, w); out[b*max_len*w + t*w + j].
// All slices pre-allocated. len(out) == B*maxLen*w, sum(lengths) == T.
func Pad(out []float32, X []float32, T, w int, lengths []int32, maxLen int) {
	for i := range out {
		out[i] = 0
	}
	srcOff := 0
	for b, length := range lengths {
		L := int(length)
		for t := 0; t < L; t++ {
			dstStart := b*maxLen*w + t*w
			srcStart := (srcOff + t) * w
			copy(out[dstStart:dstStart+w], X[srcStart:srcStart+w])
		}
		srcOff += L
	}
}
