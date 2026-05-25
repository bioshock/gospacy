package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// PrecomputableAffine is the parser's `lower` layer: it caches per-token
// feature contributions for every (token, feature_slot) pair so the per-state
// scorer can compose state vectors via O(nF) additions instead of an
// O(T*nF*nO*nP) matmul per state. Ports
// spacy.ml._precomputable_affine.PrecomputableAffine.
//
// Param layout (matches thinc on-disk):
//   - W: (nF, nO, nP, nI) flat = nF*nO*nP*nI float32
//   - b: (nO, nP) flat = nO*nP float32 (added per-state by the scorer, not here)
//   - pad: (1, nF, nO, nP) flat = nF*nO*nP float32 (filler for missing features,
//     consumed when context-token index is -1)
//
// Forward input: X (Floats2d, T × nI).
// Forward output: Floats2d with Rows=(T+1)*nF, Cols=nO*nP. The scorer
// reinterprets this as a logical (T+1, nF, nO, nP) tensor; row 0..nF-1 of the
// flat layout is the pad row (used when context-token == -1).
func PrecomputableAffine(ops nn.Ops, nO, nI, nF, nP int) *nn.Model {
	// Grow-on-demand scratch reused across Pipe calls. Safe per
	// Bundle.Pipe single-goroutine contract. zeroBias capacity caps at
	// nF*nO*nP (constant for a given layer instance) so it's allocated
	// once after the first Forward.
	var scratchOut, scratchZeroBias []float32
	m := &nn.Model{
		Name: "precomputable_affine",
		Ops:  ops,
		Params: map[string][]float32{
			"W":   nil,
			"b":   nil,
			"pad": nil,
		},
		Dims: map[string]int{
			"nF": nF,
			"nI": nI,
			"nO": nO,
			"nP": nP,
		},
		Attrs: map[string]any{},
	}
	m.Forward = func(m *nn.Model, X any) (any, error) {
		in, ok := X.(nn.Floats2d)
		if !ok {
			return nil, fmt.Errorf("PrecomputableAffine: expected Floats2d, got %T", X)
		}
		nF := m.Dims["nF"]
		nO := m.Dims["nO"]
		nP := m.Dims["nP"]
		nI := m.Dims["nI"]
		if in.Cols != nI {
			return nil, fmt.Errorf("PrecomputableAffine: input cols %d != nI %d", in.Cols, nI)
		}
		W := m.Params["W"]
		if len(W) != nF*nO*nP*nI {
			return nil, fmt.Errorf("PrecomputableAffine: W length %d != nF*nO*nP*nI %d", len(W), nF*nO*nP*nI)
		}
		pad := m.Params["pad"]
		if len(pad) != nF*nO*nP {
			return nil, fmt.Errorf("PrecomputableAffine: pad length %d != nF*nO*nP %d", len(pad), nF*nO*nP)
		}

		T := in.Rows
		flatRows := T + 1
		n := flatRows * nF * nO * nP
		if cap(scratchOut) < n {
			scratchOut = make([]float32, n)
		}
		out := nn.Floats2d{
			Data: scratchOut[:n],
			Rows: flatRows * nF,
			Cols: nO * nP,
		}

		// Row 0 of the logical (T+1, nF, nO, nP) tensor is the pad row.
		copy(out.Data[:nF*nO*nP], pad)

		M := T
		N := nF * nO * nP
		K := nI
		if M > 0 {
			dst := out.Data[nF*nO*nP:]
			if cap(scratchZeroBias) < N {
				scratchZeroBias = make([]float32, N)
			} else {
				// Zero out the prefix we'll use (in case prior call wrote anywhere).
				clear(scratchZeroBias[:N])
			}
			m.Ops.Affine(dst, in.Data, M, K, W, N, scratchZeroBias[:N])
		}
		return out, nil
	}
	return m
}
