package layers_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/bioshock/gospacy/v3/nn/layers"
)

func TestList2Array_ConcatenatesRows(t *testing.T) {
	m := layers.List2Array(gonum.New())
	in := nn.FloatList{Items: []nn.Floats2d{
		{Data: []float32{1, 2, 3, 4}, Rows: 2, Cols: 2},
		{Data: []float32{5, 6}, Rows: 1, Cols: 2},
	}}
	raw, err := m.Predict(in)
	require.NoError(t, err)
	got, ok := raw.(nn.Floats2d)
	require.True(t, ok)
	require.Equal(t, 3, got.Rows)
	require.Equal(t, 2, got.Cols)
	require.Equal(t, []float32{1, 2, 3, 4, 5, 6}, got.Data)
}
