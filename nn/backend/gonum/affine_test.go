package gonum

import (
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func TestAffine_AgainstSample(t *testing.T) {
	fx, err := diff.LoadOpFixture(goldenSamplePath(t))
	require.NoError(t, err)
	c, ok := fx.Ops["affine"]
	require.True(t, ok)

	X, _ := c.ArrayInput("X")
	W, _ := c.ArrayInput("W")
	b, _ := c.ArrayInput("b")
	m, _ := c.IntInput("m")
	k, _ := c.IntInput("k")
	n, _ := c.IntInput("n")

	out := make([]float32, m*n)
	Affine(out, X.Data, m, k, W.Data, n, b.Data)

	exp, err := c.Float32Output()
	require.NoError(t, err)
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-5, RelMax: 1e-5})
	require.Truef(t, rep.Equal(),
		"affine output mismatch (case=%q): first disagree at %d, maxAbsDiff=%g",
		c.Name, rep.FirstDisagreeIdx, rep.MaxAbsDiff)
}
