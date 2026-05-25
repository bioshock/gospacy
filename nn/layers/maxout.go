package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// Maxout: Y = max-over-pieces(reshape(X @ W.T + b, (rows, nO, nP))).
//
// nP is the number of "pieces" per output unit; W has shape (nO*nP, nI) and
// b has shape (nO*nP,). The argmax-per-row is returned as the second output
// of Ops.Maxout, but the Forward callback only returns the max values.
func Maxout(ops nn.Ops, nO, nI, nP int) *nn.Model {
	// Grow-on-demand scratch reused across Pipe calls. Safe because
	// Bundle.Pipe is single-goroutine by contract (see Issue A docs in
	// CHANGELOG Unreleased; vocab.Vocab.Get / Bundle.Pipe godoc).
	var scratchIntermediate, scratchOut []float32
	var scratchWhich []int32
	return &nn.Model{
		Name:   "maxout",
		Ops:    ops,
		Dims:   map[string]int{"nO": nO, "nI": nI, "nP": nP},
		Params: map[string][]float32{"W": nil, "b": nil},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("Maxout: expected Floats2d, got %T", X)
			}
			nO := m.Dims["nO"]
			nI := m.Dims["nI"]
			nP := m.Dims["nP"]
			if x.Cols != nI {
				return nil, fmt.Errorf("Maxout: input Cols=%d != nI=%d", x.Cols, nI)
			}
			nIm := x.Rows * nO * nP
			if cap(scratchIntermediate) < nIm {
				scratchIntermediate = make([]float32, nIm)
			}
			intermediate := scratchIntermediate[:nIm]
			m.Ops.Affine(intermediate, x.Data, x.Rows, nI, m.Params["W"], nO*nP, m.Params["b"])
			nOut := x.Rows * nO
			if cap(scratchOut) < nOut {
				scratchOut = make([]float32, nOut)
			}
			out := scratchOut[:nOut]
			if cap(scratchWhich) < nOut {
				scratchWhich = make([]int32, nOut)
			}
			which := scratchWhich[:nOut]
			m.Ops.Maxout(out, which, intermediate, x.Rows, nO, nP)
			return nn.Floats2d{Data: out, Rows: x.Rows, Cols: nO}, nil
		},
	}
}
