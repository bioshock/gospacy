package gonum

import (
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func TestMaxout_AgainstSample(t *testing.T) {
	fx, err := diff.LoadOpFixture(goldenSamplePath(t))
	require.NoError(t, err)
	c, ok := fx.Ops["maxout"]
	require.True(t, ok)

	X, _ := c.ArrayInput("X")
	n, _ := c.IntInput("n")
	h, _ := c.IntInput("h")
	p, _ := c.IntInput("p")

	out := make([]float32, n*h)
	which := make([]int32, n*h)
	Maxout(out, which, X.Data, n, h, p)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-6, RelMax: 1e-6})
	require.Truef(t, rep.Equal(),
		"maxout mismatch (case=%q): first disagree at %d, maxAbsDiff=%g",
		c.Name, rep.FirstDisagreeIdx, rep.MaxAbsDiff)
}
