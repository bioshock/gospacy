package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

// TestStaticVectors_Forward verifies the lookup-then-affine flow with a tiny
// hand-rolled W and vectors table. Covers the rebuilt nO/nM contract (Phase 7
// Block C3): vectors live in Attrs (injected by the bundle loader), W in
// Params, no bias.
func TestStaticVectors_Forward(t *testing.T) {
	ops := gonum.New()
	m := StaticVectors(ops, 2, 4)
	m.Attrs["vectors"] = []float32{
		1, 0, 0, 0, // row 0
		0, 1, 0, 0, // row 1
		0, 0, 1, 0, // row 2
	}
	m.Attrs["nV"] = 3
	// W is (nO=2, nM=4) — picks element 0 and element 2 of the input vector.
	m.Params["W"] = []float32{
		1, 0, 0, 0,
		0, 0, 1, 0,
	}

	rows := nn.Ints1d{Data: []int32{0, 2}}
	out, err := m.Predict(rows)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	require.Equal(t, 2, y.Rows)
	require.Equal(t, 2, y.Cols)
	// Row 0 lookup [1,0,0,0] · W^T = [1, 0]
	require.InDelta(t, 1.0, y.Data[0], 1e-6)
	require.InDelta(t, 0.0, y.Data[1], 1e-6)
	// Row 2 lookup [0,0,1,0] · W^T = [0, 1]
	require.InDelta(t, 0.0, y.Data[2], 1e-6)
	require.InDelta(t, 1.0, y.Data[3], 1e-6)
}

// TestStaticVectors_OOVRowZeroed verifies that row index -1 produces a zero
// output row (matching upstream's `vectors_data[rows < 0] = 0` semantics).
func TestStaticVectors_OOVRowZeroed(t *testing.T) {
	ops := gonum.New()
	m := StaticVectors(ops, 2, 4)
	m.Attrs["vectors"] = []float32{
		1, 1, 1, 1,
	}
	m.Attrs["nV"] = 1
	m.Params["W"] = []float32{
		2, 2, 2, 2,
		3, 3, 3, 3,
	}
	rows := nn.Ints1d{Data: []int32{0, -1, 0}}
	out, err := m.Predict(rows)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	require.Equal(t, 3, y.Rows)
	require.Equal(t, 2, y.Cols)
	// Row 0 in-vocab: [1,1,1,1] · W^T = [8, 12]
	require.InDelta(t, 8.0, y.Data[0], 1e-6)
	require.InDelta(t, 12.0, y.Data[1], 1e-6)
	// Row 1 OOV → zeros
	require.InDelta(t, 0.0, y.Data[2], 1e-6)
	require.InDelta(t, 0.0, y.Data[3], 1e-6)
	// Row 2 in-vocab again → same as row 0
	require.InDelta(t, 8.0, y.Data[4], 1e-6)
	require.InDelta(t, 12.0, y.Data[5], 1e-6)
}

// TestStaticVectors_OOBRejected verifies that an out-of-range row index
// surfaces as an error rather than reading garbage memory.
func TestStaticVectors_OOBRejected(t *testing.T) {
	ops := gonum.New()
	m := StaticVectors(ops, 2, 4)
	m.Attrs["vectors"] = make([]float32, 3*4)
	m.Attrs["nV"] = 3
	m.Params["W"] = make([]float32, 2*4)
	rows := nn.Ints1d{Data: []int32{5}}
	_, err := m.Predict(rows)
	require.Error(t, err)
	require.Contains(t, err.Error(), "out of range")
}
