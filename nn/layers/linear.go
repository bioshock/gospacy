// Package layers provides thinc-equivalent layer constructors. Each constructor
// returns a *nn.Model with its Forward function pre-wired.
package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// Linear is an affine projection: Y = X @ W.T + b.
//
// Inputs:
//   nO: output dim (Y.Cols)
//   nI: input dim  (X.Cols)
// Params (populated by FromBytes):
//   W: shape (nO, nI) row-major
//   b: shape (nO,)
func Linear(ops nn.Ops, nO, nI int) *nn.Model {
	return &nn.Model{
		Name:   "linear",
		Ops:    ops,
		Dims:   map[string]int{"nO": nO, "nI": nI},
		Params: map[string][]float32{"W": nil, "b": nil},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("Linear: expected Floats2d, got %T", X)
			}
			nO := m.Dims["nO"]
			nI := m.Dims["nI"]
			if x.Cols != nI {
				return nil, fmt.Errorf("Linear: input Cols=%d != nI=%d", x.Cols, nI)
			}
			W := m.Params["W"]
			b := m.Params["b"]
			if len(W) != nO*nI {
				return nil, fmt.Errorf("Linear: W has %d floats, expected %d (nO*nI)", len(W), nO*nI)
			}
			if len(b) != nO {
				return nil, fmt.Errorf("Linear: b has %d floats, expected %d", len(b), nO)
			}
			out := make([]float32, x.Rows*nO)
			m.Ops.Affine(out, x.Data, x.Rows, nI, W, nO, b)
			return nn.Floats2d{Data: out, Rows: x.Rows, Cols: nO}, nil
		},
	}
}
