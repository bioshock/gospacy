package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// List2Array converts FloatList (a list of per-doc Floats2d) into a single
// Floats2d by row-wise concatenation. Mirrors thinc.layers.list2array.list2array.
// Used by the parser's tok2vec sub-chain to flatten the listener's per-doc
// output before the 96→64 projection.
func List2Array(ops nn.Ops) *nn.Model {
	return &nn.Model{
		Name:    "list2array",
		Ops:     ops,
		Forward: list2arrayForward,
	}
}

func list2arrayForward(_ *nn.Model, X any) (any, error) {
	in, ok := X.(nn.FloatList)
	if !ok {
		return nil, fmt.Errorf("List2Array: expected FloatList, got %T", X)
	}
	if len(in.Items) == 0 {
		return nn.Floats2d{}, nil
	}
	cols := in.Items[0].Cols
	totalRows := 0
	for _, it := range in.Items {
		if it.Cols != cols {
			return nil, fmt.Errorf("List2Array: item col mismatch (%d vs %d)", it.Cols, cols)
		}
		totalRows += it.Rows
	}
	out := nn.Floats2d{
		Data: make([]float32, totalRows*cols),
		Rows: totalRows,
		Cols: cols,
	}
	offset := 0
	for _, it := range in.Items {
		copy(out.Data[offset:], it.Data)
		offset += it.Rows * cols
	}
	return out, nil
}
