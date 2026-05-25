package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// Ragged2List is the inverse of List2Ragged: split a Ragged batch back into a
// FloatList of (Lengths[i], Cols) Floats2d items by per-sequence row count.
// Mirrors thinc.layers.ragged2list (an ops.unflatten call on the data tensor).
func Ragged2List(ops nn.Ops) *nn.Model {
	return &nn.Model{
		Name: "ragged2list",
		Ops:  ops,
		Forward: func(_ *nn.Model, X any) (any, error) {
			r, ok := X.(nn.Ragged)
			if !ok {
				return nil, fmt.Errorf("Ragged2List: expected Ragged, got %T", X)
			}
			items := make([]nn.Floats2d, len(r.Lengths))
			off := 0
			for i, ln := range r.Lengths {
				rows := int(ln)
				n := rows * r.Cols
				if off+n > len(r.Data) {
					return nil, fmt.Errorf("Ragged2List: item %d wants %d floats, only %d remain",
						i, n, len(r.Data)-off)
				}
				slice := make([]float32, n)
				copy(slice, r.Data[off:off+n])
				items[i] = nn.Floats2d{Data: slice, Rows: rows, Cols: r.Cols}
				off += n
			}
			return nn.FloatList{Items: items}, nil
		},
	}
}
