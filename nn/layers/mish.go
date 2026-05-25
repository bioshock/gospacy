package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// Mish is an affine-plus-mish-activation layer: Y = mish(X @ W.T + b).
func Mish(ops nn.Ops, nO, nI int) *nn.Model {
	return &nn.Model{
		Name:   "mish",
		Ops:    ops,
		Dims:   map[string]int{"nO": nO, "nI": nI},
		Params: map[string][]float32{"W": nil, "b": nil},
		Attrs:  map[string]any{"threshold": float32(20)},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("Mish: expected Floats2d, got %T", X)
			}
			nO := m.Dims["nO"]
			nI := m.Dims["nI"]
			if x.Cols != nI {
				return nil, fmt.Errorf("Mish: input Cols=%d != nI=%d", x.Cols, nI)
			}
			out := make([]float32, x.Rows*nO)
			m.Ops.Affine(out, x.Data, x.Rows, nI, m.Params["W"], nO, m.Params["b"])
			threshold, _ := m.Attrs["threshold"].(float32)
			if threshold == 0 {
				threshold = 20
			}
			m.Ops.Mish(out, out, threshold)
			return nn.Floats2d{Data: out, Rows: x.Rows, Cols: nO}, nil
		},
	}
}
