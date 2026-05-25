package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// List2Ragged converts a list of 2D arrays to a ragged batch. Polymorphic:
//
//   - FloatList   → Ragged    (used on the encode path, where each list item is
//     a Floats2d of token embeddings for one Doc).
//   - []Uint64s2d → RaggedU64 (used on the embed path: ExtractFeatures emits
//     one Uint64s2d per Doc; the MultiHashEmbed sub-tree consumes the
//     concatenated batch via WithArray's RaggedU64 dispatch).
//
// All items must share Cols.
func List2Ragged(ops nn.Ops) *nn.Model {
	return &nn.Model{
		Name: "list2ragged",
		Ops:  ops,
		Forward: func(_ *nn.Model, X any) (any, error) {
			switch lst := X.(type) {
			case nn.FloatList:
				if len(lst.Items) == 0 {
					return nn.Ragged{Cols: 0}, nil
				}
				cols := lst.Items[0].Cols
				totalRows := 0
				lengths := make([]int32, len(lst.Items))
				for i, it := range lst.Items {
					if it.Cols != cols {
						return nil, fmt.Errorf("List2Ragged: item %d Cols=%d, expected %d", i, it.Cols, cols)
					}
					lengths[i] = int32(it.Rows)
					totalRows += it.Rows
				}
				data := make([]float32, totalRows*cols)
				off := 0
				for _, it := range lst.Items {
					copy(data[off:off+len(it.Data)], it.Data)
					off += len(it.Data)
				}
				return nn.Ragged{Data: data, Lengths: lengths, Cols: cols}, nil
			case []nn.Uint64s2d:
				if len(lst) == 0 {
					return nn.RaggedU64{Cols: 0}, nil
				}
				cols := lst[0].Cols
				totalRows := 0
				lengths := make([]int32, len(lst))
				for i, it := range lst {
					if it.Cols != cols {
						return nil, fmt.Errorf("List2Ragged: item %d Cols=%d, expected %d", i, it.Cols, cols)
					}
					lengths[i] = int32(it.Rows)
					totalRows += it.Rows
				}
				data := make([]uint64, totalRows*cols)
				off := 0
				for _, it := range lst {
					copy(data[off:off+len(it.Data)], it.Data)
					off += len(it.Data)
				}
				return nn.RaggedU64{Data: data, Lengths: lengths, Cols: cols}, nil
			default:
				return nil, fmt.Errorf("List2Ragged: expected FloatList or []Uint64s2d, got %T", X)
			}
		},
	}
}
