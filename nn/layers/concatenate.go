package layers

import (
	"fmt"
	"strings"

	"github.com/bioshock/gospacy/v3/nn"
)

// Concatenate applies each sub-model to the same input X and concatenates
// their outputs along the Cols dim. Polymorphic on the output kind, matching
// thinc.layers.concatenate's `_array_forward` / `_ragged_forward` dispatch:
//
//   - Floats2d outputs (e.g. hash-embed arms): merged into a single Floats2d
//     with the same Rows, Cols = sum(child_cols).
//   - Ragged outputs (e.g. md/lg's feature_extractor | static_vectors arm):
//     merged into a single Ragged with the same Lengths, Cols = sum(child_cols).
//
// All children must agree on the output kind and the Rows/Lengths dim. Mixed
// outputs (e.g. one Floats2d + one Ragged) fail loud at runtime.
func Concatenate(ops nn.Ops, sublayers ...*nn.Model) *nn.Model {
	names := make([]string, len(sublayers))
	for i, s := range sublayers {
		names[i] = s.Name
	}
	var scratchF []float32 // grow-on-demand; safe per Bundle.Pipe single-goroutine contract.
	var scratchR []float32 // separate buffer for Ragged path so type doesn't matter.
	return &nn.Model{
		// thinc names a Concatenate node by joining child names with "|" (no
		// wrapper). This matches `Model.__init__`'s naming convention and the
		// on-disk `tok2vec/model` payload's BFS node-name array.
		Name:   strings.Join(names, "|"),
		Ops:    ops,
		Layers: sublayers,
		Forward: func(m *nn.Model, X any) (any, error) {
			outs := make([]any, len(m.Layers))
			for i, child := range m.Layers {
				out, err := child.Predict(X)
				if err != nil {
					return nil, fmt.Errorf("Concatenate sublayer %d: %w", i, err)
				}
				outs[i] = out
			}
			switch first := outs[0].(type) {
			case nn.Floats2d:
				parts := make([]nn.Floats2d, len(outs))
				totalCols := first.Cols
				parts[0] = first
				for i := 1; i < len(outs); i++ {
					p, ok := outs[i].(nn.Floats2d)
					if !ok {
						return nil, fmt.Errorf(
							"Concatenate: sublayer %d returned %T, expected Floats2d (mixed kinds)",
							i, outs[i],
						)
					}
					if p.Rows != first.Rows {
						return nil, fmt.Errorf("Concatenate: sublayer %d rows=%d, expected %d",
							i, p.Rows, first.Rows)
					}
					parts[i] = p
					totalCols += p.Cols
				}
				n := first.Rows * totalCols
				if cap(scratchF) < n {
					scratchF = make([]float32, n)
				}
				out := scratchF[:n]
				for r := 0; r < first.Rows; r++ {
					off := 0
					for _, p := range parts {
						src := p.Data[r*p.Cols : (r+1)*p.Cols]
						copy(out[r*totalCols+off:r*totalCols+off+p.Cols], src)
						off += p.Cols
					}
				}
				return nn.Floats2d{Data: out, Rows: first.Rows, Cols: totalCols}, nil
			case nn.Ragged:
				parts := make([]nn.Ragged, len(outs))
				totalCols := first.Cols
				parts[0] = first
				totalRows := 0
				for _, l := range first.Lengths {
					totalRows += int(l)
				}
				for i := 1; i < len(outs); i++ {
					p, ok := outs[i].(nn.Ragged)
					if !ok {
						return nil, fmt.Errorf(
							"Concatenate: sublayer %d returned %T, expected Ragged (mixed kinds)",
							i, outs[i],
						)
					}
					if len(p.Lengths) != len(first.Lengths) {
						return nil, fmt.Errorf(
							"Concatenate: sublayer %d Lengths len=%d, expected %d",
							i, len(p.Lengths), len(first.Lengths),
						)
					}
					for j := range p.Lengths {
						if p.Lengths[j] != first.Lengths[j] {
							return nil, fmt.Errorf(
								"Concatenate: sublayer %d Lengths[%d]=%d, expected %d",
								i, j, p.Lengths[j], first.Lengths[j],
							)
						}
					}
					parts[i] = p
					totalCols += p.Cols
				}
				n := totalRows * totalCols
				if cap(scratchR) < n {
					scratchR = make([]float32, n)
				}
				out := scratchR[:n]
				for r := 0; r < totalRows; r++ {
					off := 0
					for _, p := range parts {
						src := p.Data[r*p.Cols : (r+1)*p.Cols]
						copy(out[r*totalCols+off:r*totalCols+off+p.Cols], src)
						off += p.Cols
					}
				}
				return nn.Ragged{Data: out, Lengths: append([]int32(nil), first.Lengths...), Cols: totalCols}, nil
			default:
				return nil, fmt.Errorf("Concatenate: sublayer 0 returned %T, expected Floats2d or Ragged", outs[0])
			}
		},
	}
}
