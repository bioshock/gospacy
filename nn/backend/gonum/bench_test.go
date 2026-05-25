package gonum

import (
	"math/rand/v2"
	"testing"
)

func randomFloat32s(n int, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}

// Gemm: 30x96 @ 96x300 — roughly the tagger's affine projection.
func BenchmarkGemm_30x96_96x300(b *testing.B) {
	A := randomFloat32s(30*96, 1)
	B_ := randomFloat32s(96*300, 2)
	out := make([]float32, 30*300)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Gemm(out, A, 30, 96, B_, 300)
	}
}

// Affine: same as gemm + bias broadcast.
func BenchmarkAffine_30x96_300x96(b *testing.B) {
	X := randomFloat32s(30*96, 1)
	W := randomFloat32s(300*96, 2)
	bias := randomFloat32s(300, 3)
	out := make([]float32, 30*300)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Affine(out, X, 30, 96, W, 300, bias)
	}
}

// Seq2Col: 30 tokens, 96 dim, window=1 — typical tok2vec input.
func BenchmarkSeq2Col_30x96_nW1(b *testing.B) {
	X := randomFloat32s(30*96, 1)
	out := make([]float32, 30*3*96)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Seq2Col(out, X, 30, 96, 1)
	}
}

// Maxout: 30 tokens, hidden 96, pieces 3 — Tok2Vec MaxoutWindowEncoder default.
func BenchmarkMaxout_30x96_p3(b *testing.B) {
	X := randomFloat32s(30*96*3, 1)
	out := make([]float32, 30*96)
	which := make([]int32, 30*96)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Maxout(out, which, X, 30, 96, 3)
	}
}

// Mish: 30 tokens * 96 dim element-wise activation.
func BenchmarkMish_30x96(b *testing.B) {
	X := randomFloat32s(30*96, 1)
	out := make([]float32, 30*96)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Mish(out, X, 20.0)
	}
}

// Softmax: 30 rows, 50 classes — typical tagger output.
func BenchmarkSoftmax_30x50(b *testing.B) {
	X := randomFloat32s(30*50, 1)
	out := make([]float32, 30*50)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Softmax(out, X, 30, 50)
	}
}

// Hash: 30 ids -> 4 hashes each.
func BenchmarkHash_30ids(b *testing.B) {
	ids := make([]uint64, 30)
	for i := range ids {
		ids[i] = uint64(i * 31337)
	}
	out := make([]uint32, 30*4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Hash(out, ids, 0)
	}
}

// GatherAdd: 30 tokens, 4 lookups, 96-dim table rows.
func BenchmarkGatherAdd_30tokens_4lookups_96dim(b *testing.B) {
	table := randomFloat32s(1000*96, 1)
	indices := make([]int32, 30*4)
	for i := range indices {
		indices[i] = int32(i % 1000)
	}
	out := make([]float32, 30*96)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GatherAdd(out, table, 1000, 96, indices, 30, 4)
	}
}
