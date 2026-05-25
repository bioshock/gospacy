package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// ExpandWindow wraps Ops.Seq2Col as a layer.
//
// Input Floats2d (n, w) → output Floats2d (n, (2*nW+1)*w).
func ExpandWindow(ops nn.Ops, nW int) *nn.Model {
	var scratch []float32 // grow-on-demand; safe per Bundle.Pipe single-goroutine contract.
	return &nn.Model{
		Name:  "expand_window",
		Ops:   ops,
		Attrs: map[string]any{"nW": nW},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("ExpandWindow: expected Floats2d, got %T", X)
			}
			nW, _ := m.Attrs["nW"].(int)
			outCols := (2*nW + 1) * x.Cols
			n := x.Rows * outCols
			if cap(scratch) < n {
				scratch = make([]float32, n)
			}
			out := scratch[:n]
			m.Ops.Seq2Col(out, x.Data, x.Rows, x.Cols, nW)
			return nn.Floats2d{Data: out, Rows: x.Rows, Cols: outCols}, nil
		},
	}
}
