package gonum

import (
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func TestSoftmax_AgainstSample(t *testing.T) {
	fx, err := diff.LoadOpFixture(goldenSamplePath(t))
	require.NoError(t, err)
	c, ok := fx.Ops["softmax"]
	require.True(t, ok)

	X, _ := c.ArrayInput("X")
	n, _ := c.IntInput("n")
	k, _ := c.IntInput("k")

	out := make([]float32, n*k)
	Softmax(out, X.Data, n, k)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-6, RelMax: 1e-5})
	require.Truef(t, rep.Equal(),
		"softmax mismatch (case=%q): first disagree at %d, maxAbsDiff=%g",
		c.Name, rep.FirstDisagreeIdx, rep.MaxAbsDiff)
}
