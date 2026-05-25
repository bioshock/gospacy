package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// withArrayPad reads the optional "pad" attribute from a WithArray node.
// Accepts int (Go-built) and int64 (config-loaded) interchangeably so the
// pad value flows through whether the model is built via Go literals or
// hydrated from a registry cfg.
func withArrayPad(m *nn.Model) int {
	if m == nil || m.Attrs == nil {
		return 0
	}
	switch v := m.Attrs["pad"].(type) {
	case int:
		return v
	case int64:
		return int(v)
	default:
		return 0
	}
}

// WithArray treats a sequence input as a flat 2D array (concat'd rows), applies
// the inner layer, and re-wraps the result preserving the per-sequence
// boundaries. The inner layer must preserve the row count.
//
// Polymorphic over the input kind (matches thinc.layers.with_array's three
// dispatch arms in `_ragged_forward` / `_list_forward`):
//
//   - Ragged    → flat Floats2d → inner → Floats2d → Ragged    (embed-reduce
//     path: the embed sub-chain feeds a Ragged of width-N×K embeddings into
//     the Maxout reducer).
//   - RaggedU64 → flat Uint64s2d → inner → Floats2d → Ragged   (embed path:
//     MultiHashEmbed maps uint64 IDs to float embeddings, so the output is
//     always float).
//   - FloatList → flat Floats2d → inner → Floats2d → FloatList (encode path:
//     Ragged2List emits a List[Floats2d]; the residual encoder reads/writes
//     the same shape).
//
// `Attrs["pad"]` is the optional zero-row pad applied on both sides of each
// list item before flattening (and stripped from the output prefix on
// unflatten). Defaults to 0. Mirrors thinc's with_array(layer, pad=N), used
// by MaxoutWindowEncoder with pad=window_size*depth so each Doc's
// ExpandWindow operations see zero-padding instead of leaking across Doc
// boundaries (the `pad` attribute is only honored on the FloatList arm; the
// Ragged / RaggedU64 arms have a single concatenated sequence with no inter-
// doc boundary). Reads as int (default), and accepts int64 (config-loaded
// path) interchangeably.
func WithArray(ops nn.Ops, inner *nn.Model) *nn.Model {
	return &nn.Model{
		Name:   "with_array(" + inner.Name + ")",
		Ops:    ops,
		Layers: []*nn.Model{inner},
		Attrs:  map[string]any{"pad": 0},
		Forward: func(m *nn.Model, X any) (any, error) {
			pad := withArrayPad(m)
			switch r := X.(type) {
			case nn.Ragged:
				totalRows := 0
				for _, l := range r.Lengths {
					totalRows += int(l)
				}
				flat := nn.Floats2d{Data: r.Data, Rows: totalRows, Cols: r.Cols}
				out, err := m.Layers[0].Predict(flat)
				if err != nil {
					return nil, err
				}
				y, ok := out.(nn.Floats2d)
				if !ok {
					return nil, fmt.Errorf("WithArray inner returned %T, want Floats2d", out)
				}
				if y.Rows != totalRows {
					return nil, fmt.Errorf("WithArray inner changed row count: %d -> %d", totalRows, y.Rows)
				}
				return nn.Ragged{Data: y.Data, Lengths: r.Lengths, Cols: y.Cols}, nil
			case nn.RaggedU64:
				totalRows := 0
				for _, l := range r.Lengths {
					totalRows += int(l)
				}
				flat := nn.Uint64s2d{Data: r.Data, Rows: totalRows, Cols: r.Cols}
				out, err := m.Layers[0].Predict(flat)
				if err != nil {
					return nil, err
				}
				y, ok := out.(nn.Floats2d)
				if !ok {
					return nil, fmt.Errorf("WithArray (uint64) inner returned %T, want Floats2d", out)
				}
				if y.Rows != totalRows {
					return nil, fmt.Errorf("WithArray (uint64) inner changed row count: %d -> %d", totalRows, y.Rows)
				}
				return nn.Ragged{Data: y.Data, Lengths: r.Lengths, Cols: y.Cols}, nil
			case nn.FloatList:
				// Flatten List[Floats2d] → Floats2d, run inner, unflatten back
				// using per-item Rows. Mirrors thinc's _list_forward.
				//
				// pad > 0: each item is wrapped in `pad` zero rows on both
				// sides before flatten; the inner is fed the padded array;
				// on unflatten, each per-item slice is `pad + Rows` long
				// (suffix is dropped via cumsum semantics) and the `pad`-row
				// prefix is then stripped. See thinc/backends/ops.py
				// flatten/unflatten and thinc/layers/with_array.py
				// _list_forward.
				if len(r.Items) == 0 {
					return r, nil
				}
				cols := r.Items[0].Cols
				lengths := make([]int, len(r.Items))
				totalRowsInner := 0
				for i, it := range r.Items {
					if it.Cols != cols {
						return nil, fmt.Errorf("WithArray (list): item %d Cols=%d, expected %d", i, it.Cols, cols)
					}
					lengths[i] = it.Rows
					// One pad-prefix + item rows + one pad-suffix per item
					// when pad > 0; no padding when pad == 0.
					if pad > 0 {
						totalRowsInner += pad + it.Rows + pad
					} else {
						totalRowsInner += it.Rows
					}
				}
				flatData := make([]float32, totalRowsInner*cols)
				off := 0
				for _, it := range r.Items {
					if pad > 0 {
						// Leading pad rows are zero (slice already zero-init'd).
						off += pad * cols
					}
					copy(flatData[off:off+len(it.Data)], it.Data)
					off += len(it.Data)
					if pad > 0 {
						// Trailing pad rows.
						off += pad * cols
					}
				}
				flat := nn.Floats2d{Data: flatData, Rows: totalRowsInner, Cols: cols}
				out, err := m.Layers[0].Predict(flat)
				if err != nil {
					return nil, err
				}
				y, ok := out.(nn.Floats2d)
				if !ok {
					return nil, fmt.Errorf("WithArray (list) inner returned %T, want Floats2d", out)
				}
				if y.Rows != totalRowsInner {
					return nil, fmt.Errorf("WithArray (list) inner changed row count: %d -> %d", totalRowsInner, y.Rows)
				}
				items := make([]nn.Floats2d, len(r.Items))
				off = 0
				for i, n := range lengths {
					if pad > 0 {
						// Thinc unflatten splits at cumsum(n + pad), keeping the
						// pad-prefix (suffix discarded by [:-1]), then strips
						// the pad-prefix. Net: skip pad rows, take n rows,
						// skip pad rows (trailing).
						off += pad * y.Cols
					}
					slice := make([]float32, n*y.Cols)
					copy(slice, y.Data[off:off+n*y.Cols])
					items[i] = nn.Floats2d{Data: slice, Rows: n, Cols: y.Cols}
					off += n * y.Cols
					if pad > 0 {
						off += pad * y.Cols
					}
				}
				return nn.FloatList{Items: items}, nil
			default:
				return nil, fmt.Errorf("WithArray: expected Ragged, RaggedU64, or FloatList, got %T", X)
			}
		},
	}
}
