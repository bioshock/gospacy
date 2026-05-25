package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestWithArray_Forward(t *testing.T) {
	ops := gonum.New()
	inner := Linear(ops, 2, 2)
	inner.Params["W"] = []float32{1, 0, 0, 1}
	inner.Params["b"] = []float32{0, 0}

	m := WithArray(ops, inner)

	X := nn.Ragged{
		Data:    []float32{1, 2, 3, 4, 5, 6},
		Lengths: []int32{1, 2},
		Cols:    2,
	}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y, ok := out.(nn.Ragged)
	require.True(t, ok)
	require.Equal(t, []int32{1, 2}, y.Lengths)
	require.Equal(t, 2, y.Cols)
	require.Equal(t, X.Data, y.Data)
}
