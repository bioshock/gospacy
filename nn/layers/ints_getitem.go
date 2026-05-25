package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// IntsGetitem corresponds to thinc's ints_getitem((slice(0, None), column)).
// Input Uint64s2d (N, K) → output Uint64s1d (length N), selecting the column-
// `c` slice. Mirrors the row-`c` slice of every Doc's per-token attribute
// vector (see ExtractFeatures forward in registry/feature_extractor.go).
func IntsGetitem(ops nn.Ops, column int) *nn.Model {
	return &nn.Model{
		Name:  "ints-getitem",
		Ops:   ops,
		Attrs: map[string]any{"column": column},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Uint64s2d)
			if !ok {
				return nil, fmt.Errorf("IntsGetitem: expected Uint64s2d, got %T", X)
			}
			col, _ := m.Attrs["column"].(int)
			if col < 0 || col >= x.Cols {
				return nil, fmt.Errorf("IntsGetitem: column %d out of range [0, %d)", col, x.Cols)
			}
			out := make([]uint64, x.Rows)
			for r := 0; r < x.Rows; r++ {
				out[r] = x.Data[r*x.Cols+col]
			}
			return nn.Uint64s1d{Data: out}, nil
		},
	}
}
