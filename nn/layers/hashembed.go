package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// HashEmbed maps uint64 IDs to nO-dim embeddings via 4 hashed table lookups.
func HashEmbed(ops nn.Ops, nO, nV int, seed uint32) *nn.Model {
	// Grow-on-demand scratch reused across Pipe calls. Safe per
	// Bundle.Pipe single-goroutine contract.
	var scratchHashes []uint32
	var scratchIndices []int32
	var scratchOut []float32
	return &nn.Model{
		Name:   "hashembed",
		Ops:    ops,
		Dims:   map[string]int{"nO": nO, "nV": nV},
		Params: map[string][]float32{"E": nil},
		Attrs: map[string]any{
			"seed":         seed,
			"column":       -1,
			"dropout_rate": float32(0),
		},
		Forward: func(m *nn.Model, X any) (any, error) {
			ids, ok := X.(nn.Uint64s1d)
			if !ok {
				return nil, fmt.Errorf("HashEmbed: expected Uint64s1d, got %T", X)
			}
			nO := m.Dims["nO"]
			nV := m.Dims["nV"]
			seed, _ := m.Attrs["seed"].(uint32)
			E := m.Params["E"]
			if len(E) != nV*nO {
				return nil, fmt.Errorf("HashEmbed: E has %d floats, expected nV*nO=%d", len(E), nV*nO)
			}
			N := len(ids.Data)
			n4 := N * 4
			if cap(scratchHashes) < n4 {
				scratchHashes = make([]uint32, n4)
			}
			hashes := scratchHashes[:n4]
			m.Ops.Hash(hashes, ids.Data, seed)
			if cap(scratchIndices) < n4 {
				scratchIndices = make([]int32, n4)
			}
			indices := scratchIndices[:n4]
			nVu32 := uint32(nV)
			for i, h := range hashes {
				indices[i] = int32(h % nVu32)
			}
			nOut := N * nO
			if cap(scratchOut) < nOut {
				scratchOut = make([]float32, nOut)
			}
			out := scratchOut[:nOut]
			// GatherAdd ACCUMULATES into out; we must zero it for correctness on reuse.
			clear(out)
			m.Ops.GatherAdd(out, E, nV, nO, indices, N, 4)
			return nn.Floats2d{Data: out, Rows: N, Cols: nO}, nil
		},
	}
}
