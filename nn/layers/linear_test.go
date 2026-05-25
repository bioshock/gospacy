package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestLinear_Forward(t *testing.T) {
	ops := gonum.New()
	m := Linear(ops, 2, 3)
	m.Params["W"] = []float32{1, 0, 0, 0, 1, 0}
	m.Params["b"] = []float32{10, 20}

	X := nn.Floats2d{Data: []float32{1, 2, 3}, Rows: 1, Cols: 3}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y, ok := out.(nn.Floats2d)
	require.True(t, ok)
	require.Equal(t, 1, y.Rows)
	require.Equal(t, 2, y.Cols)
	require.InDelta(t, 11.0, y.Data[0], 1e-6)
	require.InDelta(t, 22.0, y.Data[1], 1e-6)
}

func TestLinear_DimsAndParams(t *testing.T) {
	ops := gonum.New()
	m := Linear(ops, 4, 3)
	require.Equal(t, 4, m.Dims["nO"])
	require.Equal(t, 3, m.Dims["nI"])
	_, hasW := m.Params["W"]
	_, hasB := m.Params["b"]
	require.True(t, hasW)
	require.True(t, hasB)
}
