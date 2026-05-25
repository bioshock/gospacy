package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestMaxout_Forward(t *testing.T) {
	ops := gonum.New()
	m := Maxout(ops, 1, 2, 2)
	m.Params["W"] = []float32{1, 0, 0, 1}
	m.Params["b"] = []float32{0, 0}

	X := nn.Floats2d{Data: []float32{3, 5}, Rows: 1, Cols: 2}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	require.Equal(t, 1, y.Rows)
	require.Equal(t, 1, y.Cols)
	require.InDelta(t, 5.0, y.Data[0], 1e-6)
}
