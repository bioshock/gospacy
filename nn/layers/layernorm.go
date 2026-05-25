package layers

import (
	"fmt"
	"math"

	"github.com/bioshock/gospacy/v3/nn"
)

// LayerNorm: row-wise normalise X to zero mean / unit variance, then scale by
// G and shift by b. thinc's LayerNorm.v1 uses eps=1e-8 (NOT 1e-5) — see
// thinc/layers/layernorm.py _get_moments.
//
//	Y[i,j] = ((X[i,j] - mean_i) / sqrt(var_i + 1e-8)) * G[j] + b[j]
//
// G and b are both length nI; both are filled by FromBytes at load time.
func LayerNorm(ops nn.Ops, nI int) *nn.Model {
	var scratch []float32 // grow-on-demand; safe per Bundle.Pipe single-goroutine contract.
	return &nn.Model{
		Name:   "layernorm",
		Ops:    ops,
		Dims:   map[string]int{"nI": nI, "nO": nI},
		Params: map[string][]float32{"G": nil, "b": nil},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("LayerNorm: expected Floats2d, got %T", X)
			}
			nI := m.Dims["nI"]
			if x.Cols != nI {
				return nil, fmt.Errorf("LayerNorm: input Cols=%d != nI=%d", x.Cols, nI)
			}
			G := m.Params["G"]
			b := m.Params["b"]
			if len(G) != nI {
				return nil, fmt.Errorf("LayerNorm: G has %d floats, expected nI=%d", len(G), nI)
			}
			if len(b) != nI {
				return nil, fmt.Errorf("LayerNorm: b has %d floats, expected nI=%d", len(b), nI)
			}
			n := x.Rows * x.Cols
			if cap(scratch) < n {
				scratch = make([]float32, n)
			}
			out := scratch[:n]
			eps := float32(1e-8)
			for r := 0; r < x.Rows; r++ {
				row := x.Data[r*x.Cols : (r+1)*x.Cols]
				var sum float64
				for _, v := range row {
					sum += float64(v)
				}
				mean := float32(sum / float64(x.Cols))
				var sq float64
				for _, v := range row {
					d := float64(v - mean)
					sq += d * d
				}
				variance := float32(sq / float64(x.Cols))
				invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))
				dst := out[r*x.Cols : (r+1)*x.Cols]
				for j, v := range row {
					dst[j] = ((v-mean)*invStd)*G[j] + b[j]
				}
			}
			return nn.Floats2d{Data: out, Rows: x.Rows, Cols: x.Cols}, nil
		},
	}
}
