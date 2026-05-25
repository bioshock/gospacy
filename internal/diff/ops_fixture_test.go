package diff

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLoadOpFixture_Gemm(t *testing.T) {
	root := repoRoot(t)
	fx, err := LoadOpFixture(filepath.Join(root, "testdata", "golden", "sample_ops.json"))
	require.NoError(t, err)
	gemm, ok := fx.Ops["gemm"]
	require.True(t, ok)
	require.Equal(t, "tiny_2x2_2x2", gemm.Name)

	// Decode the float32 output and verify the math.
	out, err := gemm.Float32Output()
	require.NoError(t, err)
	require.Equal(t, []int{2, 2}, out.Shape)
	require.Equal(t, "float32", out.Dtype)
	// Tiny 2x2 @ 2x2 = [[1,2],[3,4]]@[[5,6],[7,8]] = [[19,22],[43,50]]
	require.InDelta(t, 19.0, out.Data[0], 1e-6)
	require.InDelta(t, 22.0, out.Data[1], 1e-6)
	require.InDelta(t, 43.0, out.Data[2], 1e-6)
	require.InDelta(t, 50.0, out.Data[3], 1e-6)
}
