package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestConcatenate_Forward(t *testing.T) {
	ops := gonum.New()
	a := Linear(ops, 2, 2)
	a.Params["W"] = []float32{1, 0, 0, 1}
	a.Params["b"] = []float32{0, 0}
	b := Linear(ops, 3, 2)
	b.Params["W"] = []float32{1, 1, 1, 1, 1, 1}
	b.Params["b"] = []float32{1, 1, 1}

	m := Concatenate(ops, a, b)
	X := nn.Floats2d{Data: []float32{1, 2}, Rows: 1, Cols: 2}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	require.Equal(t, 1, y.Rows)
	require.Equal(t, 5, y.Cols)
	require.Equal(t, []float32{1, 2, 4, 4, 4}, y.Data)
}
