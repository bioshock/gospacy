package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

// identityPadded is a stub model that returns its Padded input unchanged.
func identityPadded(ops nn.Ops) *nn.Model {
	return &nn.Model{
		Name: "identity_padded",
		Ops:  ops,
		Forward: func(_ *nn.Model, X any) (any, error) {
			return X, nil
		},
	}
}

func TestWithPadded_Roundtrip(t *testing.T) {
	ops := gonum.New()
	m := WithPadded(ops, identityPadded(ops))

	X := nn.Ragged{
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
		Lengths: []int32{2, 4, 1},
		Cols:    2,
	}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y, ok := out.(nn.Ragged)
	require.True(t, ok)
	require.Equal(t, X.Lengths, y.Lengths)
	require.Equal(t, X.Cols, y.Cols)
	require.Equal(t, X.Data, y.Data)
}
