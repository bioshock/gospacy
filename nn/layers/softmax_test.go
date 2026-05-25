package layers

import (
	"math"
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestSoftmax_Forward(t *testing.T) {
	ops := gonum.New()
	m := Softmax(ops, 2, 2)
	m.Params["W"] = []float32{1, 0, 0, 1}
	m.Params["b"] = []float32{0, 0}

	X := nn.Floats2d{Data: []float32{1, 2}, Rows: 1, Cols: 2}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	sum := math.Exp(1) + math.Exp(2)
	require.InDelta(t, math.Exp(1)/sum, y.Data[0], 1e-5)
	require.InDelta(t, math.Exp(2)/sum, y.Data[1], 1e-5)
}
