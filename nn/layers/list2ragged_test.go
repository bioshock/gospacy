package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestList2Ragged_Forward(t *testing.T) {
	m := List2Ragged(gonum.New())
	X := nn.FloatList{Items: []nn.Floats2d{
		{Data: []float32{1, 2, 3, 4}, Rows: 2, Cols: 2},
		{Data: []float32{5, 6}, Rows: 1, Cols: 2},
	}}
	out, err := m.Predict(X)
	require.NoError(t, err)
	r := out.(nn.Ragged)
	require.Equal(t, []int32{2, 1}, r.Lengths)
	require.Equal(t, 2, r.Cols)
	require.Equal(t, []float32{1, 2, 3, 4, 5, 6}, r.Data)
}
