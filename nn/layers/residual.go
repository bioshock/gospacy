package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// Residual: Y = inner(X) + X. Input and inner output must be Floats2d of identical shape.
func Residual(ops nn.Ops, inner *nn.Model) *nn.Model {
	var scratch []float32 // grow-on-demand; safe per Bundle.Pipe single-goroutine contract.
	return &nn.Model{
		Name:   "residual(" + inner.Name + ")",
		Ops:    ops,
		Layers: []*nn.Model{inner},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("Residual: expected Floats2d, got %T", X)
			}
			out, err := m.Layers[0].Predict(x)
			if err != nil {
				return nil, err
			}
			y, ok := out.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("Residual inner returned %T, want Floats2d", out)
			}
			if y.Rows != x.Rows || y.Cols != x.Cols {
				return nil, fmt.Errorf("Residual: inner output (%d,%d) != input (%d,%d)",
					y.Rows, y.Cols, x.Rows, x.Cols)
			}
			n := len(x.Data)
			if cap(scratch) < n {
				scratch = make([]float32, n)
			}
			sum := scratch[:n]
			for i := range sum {
				sum[i] = x.Data[i] + y.Data[i]
			}
			return nn.Floats2d{Data: sum, Rows: x.Rows, Cols: x.Cols}, nil
		},
	}
}
