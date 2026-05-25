package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestChain_Forward(t *testing.T) {
	ops := gonum.New()
	linear := Linear(ops, 2, 2)
	linear.Params["W"] = []float32{1, 0, 0, 1}
	linear.Params["b"] = []float32{1, 1}

	soft := Softmax(ops, 2, 2)
	soft.Params["W"] = []float32{1, 0, 0, 1}
	soft.Params["b"] = []float32{0, 0}

	m := Chain(ops, linear, soft)
	X := nn.Floats2d{Data: []float32{0, 0}, Rows: 1, Cols: 2}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	require.Equal(t, 1, y.Rows)
	require.Equal(t, 2, y.Cols)
	require.InDelta(t, 0.5, y.Data[0], 1e-6)
	require.InDelta(t, 0.5, y.Data[1], 1e-6)
}

func TestChain_WalkOrder(t *testing.T) {
	a := &nn.Model{Name: "A"}
	b := &nn.Model{Name: "B"}
	c := &nn.Model{Name: "C"}
	m := Chain(nil, a, b, c)
	require.Equal(t, []string{"A>>B>>C", "A", "B", "C"}, walkNames(m))
}

func walkNames(m *nn.Model) []string {
	out := make([]string, 0)
	for _, w := range m.Walk() {
		out = append(out, w.Name)
	}
	return out
}
