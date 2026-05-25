package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestResidual_Forward(t *testing.T) {
	ops := gonum.New()
	inner := Linear(ops, 2, 2)
	inner.Params["W"] = []float32{1, 0, 0, 1}
	inner.Params["b"] = []float32{0, 0}

	m := Residual(ops, inner)
	X := nn.Floats2d{Data: []float32{3, 5}, Rows: 1, Cols: 2}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	require.Equal(t, []float32{6, 10}, y.Data)
}
