package gonum

import "github.com/bioshock/gospacy/v3/internal/murmur"

// Hash computes 4 hashes per uint64 id, mirroring thinc's HashEmbed
// `ops.hash(ids, seed)` which returns a (N, 4) uint32 array. Uses the
// custom 64-bit mixing routine in internal/murmur (matches thinc's
// MurmurHash3_x86_128_uint64 inline function from numpy_ops.pyx).
//
// Inputs:
//
//	ids:  length N uint64 array.
//	seed: 32-bit seed.
//
// Output:
//
//	out:  length N*4 uint32 array; out[i*4..i*4+4] are the 4 hashes for ids[i].
func Hash(out []uint32, ids []uint64, seed uint32) {
	for i, id := range ids {
		h := murmur.Hash3X86_128_Uint64(id, seed)
		base := i * 4
		out[base] = h[0]
		out[base+1] = h[1]
		out[base+2] = h[2]
		out[base+3] = h[3]
	}
}
