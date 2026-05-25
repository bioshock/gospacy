package diff

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompareFloats_Equal(t *testing.T) {
	want := []float32{1, 2, 3, 4}
	got := []float32{1, 2, 3, 4}
	r := CompareFloats(want, got, Tolerance{AbsMax: 1e-6, RelMax: 1e-6})
	require.True(t, r.Equal())
	require.Equal(t, float32(0), r.MaxAbsDiff)
	require.Equal(t, float32(0), r.MaxRelDiff)
	require.Equal(t, -1, r.FirstDisagreeIdx)
}

func TestCompareFloats_WithinTolerance(t *testing.T) {
	want := []float32{1, 2, 3}
	got := []float32{1.0000001, 2.0000001, 2.9999999} // ~1e-7 differences
	r := CompareFloats(want, got, Tolerance{AbsMax: 1e-5, RelMax: 1e-5})
	require.True(t, r.Equal())
	require.True(t, r.MaxAbsDiff < 1e-5)
}

func TestCompareFloats_OutsideTolerance(t *testing.T) {
	want := []float32{1, 2, 3}
	got := []float32{1, 2.5, 3} // 0.5 at index 1
	r := CompareFloats(want, got, Tolerance{AbsMax: 1e-3, RelMax: 1e-3})
	require.False(t, r.Equal())
	require.Equal(t, 1, r.FirstDisagreeIdx)
	require.InDelta(t, 0.5, r.MaxAbsDiff, 1e-6)
}

func TestCompareFloats_LengthMismatch(t *testing.T) {
	want := []float32{1, 2, 3}
	got := []float32{1, 2}
	r := CompareFloats(want, got, Tolerance{AbsMax: 1e-6, RelMax: 1e-6})
	require.False(t, r.Equal())
	require.NotEmpty(t, r.LengthMismatch)
}

func TestCompareFloats_NaNInGot(t *testing.T) {
	want := []float32{1, 2, 3}
	got := []float32{1, float32(math.NaN()), 3}
	r := CompareFloats(want, got, Tolerance{AbsMax: 1e-6, RelMax: 1e-6})
	require.False(t, r.Equal())
	require.True(t, r.HasNaN)
	require.Equal(t, 1, r.FirstDisagreeIdx)
}

func TestCompareFloats_ZeroExpectedAvoidsDivByZero(t *testing.T) {
	want := []float32{0, 0, 0}
	got := []float32{0, 0, 1e-7}
	r := CompareFloats(want, got, Tolerance{AbsMax: 1e-5, RelMax: 1e-5})
	// Relative diff is undefined when want=0; we should fall back to abs check.
	require.True(t, r.Equal())
}
