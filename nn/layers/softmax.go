package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// Softmax is a softmax-projection layer: Y = softmax(X @ W.T + b).
//
// thinc names this `softmax_v2` when normalize_outputs=True (default for
// classifier heads). This implementation always normalises.
func Softmax(ops nn.Ops, nO, nI int) *nn.Model {
	return &nn.Model{
		Name:   "softmax_v2",
		Ops:    ops,
		Dims:   map[string]int{"nO": nO, "nI": nI},
		Params: map[string][]float32{"W": nil, "b": nil},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("Softmax: expected Floats2d, got %T", X)
			}
			nO := m.Dims["nO"]
			nI := m.Dims["nI"]
			if x.Cols != nI {
				return nil, fmt.Errorf("Softmax: input Cols=%d != nI=%d", x.Cols, nI)
			}
			out := make([]float32, x.Rows*nO)
			m.Ops.Affine(out, x.Data, x.Rows, nI, m.Params["W"], nO, m.Params["b"])
			m.Ops.Softmax(out, out, x.Rows, nO)
			return nn.Floats2d{Data: out, Rows: x.Rows, Cols: nO}, nil
		},
	}
}
