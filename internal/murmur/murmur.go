// Package murmur implements the hash primitives used by spaCy and thinc:
//   - Hash64A: MurmurHash64A (MurmurHash2 64-bit variant A). Matches Python's
//     murmurhash.mrmr.hash64, used by spacy.strings.StringStore.
//   - Hash3X86_128_Uint64: MurmurHash3 x86_128 applied to a single uint64.
//     Matches thinc's MurmurHash3_x86_128_uint64 used by HashEmbed.
//
// Both implementations are cross-verified against the Python murmurhash package
// and thinc via golden test vectors in testdata/golden/murmur_vectors.json.
package murmur

import (
	"encoding/binary"
	"math/bits"
)

// ---- x86_32 building block used by the x86_128 variant ----

const (
	c1_x86 uint32 = 0x239b961b
	c2_x86 uint32 = 0xab0e9789
	c3_x86 uint32 = 0x38b34ae5
	c4_x86 uint32 = 0xa1e38b93
)

// Hash3X86_128_Uint64 is thinc's MurmurHash3_x86_128_uint64 inline function,
// used by HashEmbed. It is NOT the standard MurmurHash3_x86_128 applied to 8
// bytes; it is a custom 64-bit routine that combines the key with x64 mixing
// constants and MurmurHash3_x64_128 finalisers. Cross-verified against thinc's
// numpy_ops.pyx implementation and testdata/golden/murmur_vectors.json.
func Hash3X86_128_Uint64(key uint64, seed uint32) [4]uint32 {
	const (
		c1 uint64 = 0x87c37b91114253d5
		c2 uint64 = 0x4cf5ad432745937f
		f1 uint64 = 0xff51afd7ed558ccd
		f2 uint64 = 0xc4ceb9fe1a85ec53
	)

	h1 := key
	h1 *= c1
	h1 = bits.RotateLeft64(h1, 31)
	h1 *= c2
	h1 ^= uint64(seed)
	h1 ^= 8
	h2 := uint64(seed) ^ 8

	h1 += h2
	h2 += h1

	h1 ^= h1 >> 33
	h1 *= f1
	h1 ^= h1 >> 33
	h1 *= f2
	h1 ^= h1 >> 33

	h2 ^= h2 >> 33
	h2 *= f1
	h2 ^= h2 >> 33
	h2 *= f2
	h2 ^= h2 >> 33

	h1 += h2
	h2 += h1

	return [4]uint32{
		uint32(h1),
		uint32(h1 >> 32),
		uint32(h2),
		uint32(h2 >> 32),
	}
}

// Hash3X86_128 is the public MurmurHash3_x86_128 algorithm.
// Returns 4 uint32 hash values (128 bits total).
func Hash3X86_128(data []byte, seed uint32) [4]uint32 {
	h1, h2, h3, h4 := seed, seed, seed, seed
	nblocks := len(data) / 16

	// Body: 16-byte blocks
	for i := 0; i < nblocks; i++ {
		off := i * 16
		k1 := binary.LittleEndian.Uint32(data[off:])
		k2 := binary.LittleEndian.Uint32(data[off+4:])
		k3 := binary.LittleEndian.Uint32(data[off+8:])
		k4 := binary.LittleEndian.Uint32(data[off+12:])

		k1 *= c1_x86
		k1 = bits.RotateLeft32(k1, 15)
		k1 *= c2_x86
		h1 ^= k1
		h1 = bits.RotateLeft32(h1, 19)
		h1 += h2
		h1 = h1*5 + 0x561ccd1b

		k2 *= c2_x86
		k2 = bits.RotateLeft32(k2, 16)
		k2 *= c3_x86
		h2 ^= k2
		h2 = bits.RotateLeft32(h2, 17)
		h2 += h3
		h2 = h2*5 + 0x0bcaa747

		k3 *= c3_x86
		k3 = bits.RotateLeft32(k3, 17)
		k3 *= c4_x86
		h3 ^= k3
		h3 = bits.RotateLeft32(h3, 15)
		h3 += h4
		h3 = h3*5 + 0x96cd1c35

		k4 *= c4_x86
		k4 = bits.RotateLeft32(k4, 18)
		k4 *= c1_x86
		h4 ^= k4
		h4 = bits.RotateLeft32(h4, 13)
		h4 += h1
		h4 = h4*5 + 0x32ac3b17
	}

	// Tail: bytes after the last 16-byte block
	tail := data[nblocks*16:]
	var k1, k2, k3, k4 uint32
	switch len(tail) {
	case 15:
		k4 ^= uint32(tail[14]) << 16
		fallthrough
	case 14:
		k4 ^= uint32(tail[13]) << 8
		fallthrough
	case 13:
		k4 ^= uint32(tail[12])
		k4 *= c4_x86
		k4 = bits.RotateLeft32(k4, 18)
		k4 *= c1_x86
		h4 ^= k4
		fallthrough
	case 12:
		k3 ^= uint32(tail[11]) << 24
		fallthrough
	case 11:
		k3 ^= uint32(tail[10]) << 16
		fallthrough
	case 10:
		k3 ^= uint32(tail[9]) << 8
		fallthrough
	case 9:
		k3 ^= uint32(tail[8])
		k3 *= c3_x86
		k3 = bits.RotateLeft32(k3, 17)
		k3 *= c4_x86
		h3 ^= k3
		fallthrough
	case 8:
		k2 ^= uint32(tail[7]) << 24
		fallthrough
	case 7:
		k2 ^= uint32(tail[6]) << 16
		fallthrough
	case 6:
		k2 ^= uint32(tail[5]) << 8
		fallthrough
	case 5:
		k2 ^= uint32(tail[4])
		k2 *= c2_x86
		k2 = bits.RotateLeft32(k2, 16)
		k2 *= c3_x86
		h2 ^= k2
		fallthrough
	case 4:
		k1 ^= uint32(tail[3]) << 24
		fallthrough
	case 3:
		k1 ^= uint32(tail[2]) << 16
		fallthrough
	case 2:
		k1 ^= uint32(tail[1]) << 8
		fallthrough
	case 1:
		k1 ^= uint32(tail[0])
		k1 *= c1_x86
		k1 = bits.RotateLeft32(k1, 15)
		k1 *= c2_x86
		h1 ^= k1
	}

	// Finalisation
	n := uint32(len(data))
	h1 ^= n
	h2 ^= n
	h3 ^= n
	h4 ^= n
	h1 += h2
	h1 += h3
	h1 += h4
	h2 += h1
	h3 += h1
	h4 += h1
	h1 = fmix32(h1)
	h2 = fmix32(h2)
	h3 = fmix32(h3)
	h4 = fmix32(h4)
	h1 += h2
	h1 += h3
	h1 += h4
	h2 += h1
	h3 += h1
	h4 += h1
	return [4]uint32{h1, h2, h3, h4}
}

func fmix32(h uint32) uint32 {
	h ^= h >> 16
	h *= 0x85ebca6b
	h ^= h >> 13
	h *= 0xc2b2ae35
	h ^= h >> 16
	return h
}

// ---- MurmurHash64A used by spaCy StringStore via murmurhash.mrmr.hash64 ----
//
// This is the algorithm as compiled in the murmurhash Python package
// (murmurhash.mrmr.hash64 → MurmurHash64A from MurmurHash2.cpp).
// Verified by disassembling the compiled .so; two details differ from the
// canonical C reference listing in smhasher:
//   1. The mixing constant is 0xc6a4a7935bd1e995 (not 0xc6a4a7935bd064dc).
//   2. The tail XOR is reversed: k = tail[0] ^ h (not h ^= k; h *= m).
// Cross-verified against Python ctypes calls and spaCy StringStore output.

const m64a uint64 = 0xc6a4a7935bd1e995

// Hash64A computes MurmurHash64A over data using the given 32-bit seed.
// Matches Python's murmurhash.mrmr.hash64(key, seed) used by spaCy's
// StringStore. The seed is zero-extended to uint64 before hashing.
func Hash64A(data []byte, seed uint32) uint64 {
	n := len(data)
	h := uint64(seed) ^ (uint64(n) * m64a)

	// Body: consume 8 bytes at a time (little-endian).
	nblocks := n / 8
	for i := 0; i < nblocks; i++ {
		k := binary.LittleEndian.Uint64(data[i*8:])
		k *= m64a
		k ^= k >> 47
		k *= m64a
		h ^= k
		h *= m64a
	}

	// Tail: up to 7 remaining bytes.
	// The high bytes (7..1) are XORed directly into h.
	// The lowest byte is then combined as k = tail[0] ^ h; h = k * m.
	tail := data[nblocks*8:]
	switch len(tail) {
	case 7:
		h ^= uint64(tail[6]) << 48
		fallthrough
	case 6:
		h ^= uint64(tail[5]) << 40
		fallthrough
	case 5:
		h ^= uint64(tail[4]) << 32
		fallthrough
	case 4:
		h ^= uint64(tail[3]) << 24
		fallthrough
	case 3:
		h ^= uint64(tail[2]) << 16
		fallthrough
	case 2:
		h ^= uint64(tail[1]) << 8
		fallthrough
	case 1:
		k := uint64(tail[0]) ^ h // reversed: k absorbs h, not h ^= k
		h = k * m64a
	}

	// Finalisation
	h ^= h >> 47
	h *= m64a
	h ^= h >> 47
	return h
}
