package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// WithPadded converts Ragged → Padded via Ops.List2Padded, applies the inner
// layer (which takes and returns Padded), and converts back via Padded2List.
func WithPadded(ops nn.Ops, inner *nn.Model) *nn.Model {
	return &nn.Model{
		Name:   "with_padded(" + inner.Name + ")",
		Ops:    ops,
		Layers: []*nn.Model{inner},
		Forward: func(m *nn.Model, X any) (any, error) {
			r, ok := X.(nn.Ragged)
			if !ok {
				return nil, fmt.Errorf("WithPadded: expected Ragged, got %T", X)
			}
			B := len(r.Lengths)
			maxLen := 0
			totalRows := 0
			for _, l := range r.Lengths {
				if int(l) > maxLen {
					maxLen = int(l)
				}
				totalRows += int(l)
			}
			padded := nn.Padded{
				Data:    make([]float32, maxLen*B*r.Cols),
				SizeAtT: make([]int32, maxLen),
				Lengths: make([]int32, B),
				Indices: make([]int32, B),
				B:       B,
				T:       maxLen,
				W:       r.Cols,
			}
			m.Ops.List2Padded(padded.Data, padded.SizeAtT, padded.Lengths, padded.Indices,
				r.Data, totalRows, r.Cols, r.Lengths, maxLen)

			out, err := m.Layers[0].Predict(padded)
			if err != nil {
				return nil, err
			}
			yp, ok := out.(nn.Padded)
			if !ok {
				return nil, fmt.Errorf("WithPadded inner returned %T, want Padded", out)
			}
			result := nn.Ragged{
				Data:    make([]float32, totalRows*yp.W),
				Lengths: r.Lengths,
				Cols:    yp.W,
			}
			m.Ops.Padded2List(result.Data, yp.Data, yp.SizeAtT, yp.Lengths, yp.Indices,
				yp.B, yp.T, yp.W, r.Lengths)
			return result, nil
		},
	}
}
