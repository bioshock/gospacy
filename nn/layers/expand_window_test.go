package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestExpandWindow_Forward(t *testing.T) {
	m := ExpandWindow(gonum.New(), 1)
	X := nn.Floats2d{Data: []float32{1, 2, 3, 4, 5, 6}, Rows: 3, Cols: 2}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	require.Equal(t, 3, y.Rows)
	require.Equal(t, 6, y.Cols)
	require.Equal(t, []float32{0, 0, 1, 2, 3, 4}, y.Data[:6])
	require.Equal(t, []float32{1, 2, 3, 4, 5, 6}, y.Data[6:12])
	require.Equal(t, []float32{3, 4, 5, 6, 0, 0}, y.Data[12:18])
}
