package nn

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestGonumImplementsOps(t *testing.T) {
	var o Ops = gonum.New()
	require.NotNil(t, o)

	// Basic smoke: gemm 2x2 @ 2x2.
	A := []float32{1, 0, 0, 1}
	B := []float32{2, 3, 4, 5}
	out := make([]float32, 4)
	o.Gemm(out, A, 2, 2, B, 2)
	require.Equal(t, []float32{2, 3, 4, 5}, out)
}
