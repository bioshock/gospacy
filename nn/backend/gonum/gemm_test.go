package gonum

import (
	"path/filepath"
	"runtime"
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func goldenSamplePath(t *testing.T) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "testdata", "golden", "sample_ops.json")
}

func TestGemm_AgainstSample(t *testing.T) {
	fx, err := diff.LoadOpFixture(goldenSamplePath(t))
	require.NoError(t, err)
	gemmCase, ok := fx.Ops["gemm"]
	require.True(t, ok, "sample_ops.json must contain a gemm case")

	A, err := gemmCase.ArrayInput("A")
	require.NoError(t, err)
	B, err := gemmCase.ArrayInput("B")
	require.NoError(t, err)
	m, _ := gemmCase.IntInput("m")
	k, _ := gemmCase.IntInput("k")
	n, _ := gemmCase.IntInput("n")

	out := make([]float32, m*n)
	Gemm(out, A.Data, m, k, B.Data, n)

	exp, err := gemmCase.Float32Output()
	require.NoError(t, err)
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-5, RelMax: 1e-5})
	require.Truef(t, rep.Equal(),
		"gemm output mismatch (case=%q): first disagree at %d, maxAbsDiff=%g",
		gemmCase.Name, rep.FirstDisagreeIdx, rep.MaxAbsDiff)
}
