package gonum

import (
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func TestMish_AgainstSample(t *testing.T) {
	fx, err := diff.LoadOpFixture(goldenSamplePath(t))
	require.NoError(t, err)
	c, ok := fx.Ops["mish"]
	require.True(t, ok)

	X, _ := c.ArrayInput("X")
	out := make([]float32, len(X.Data))
	Mish(out, X.Data, 20.0)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-5, RelMax: 1e-5})
	require.Truef(t, rep.Equal(),
		"mish mismatch (case=%q): first disagree at %d, maxAbsDiff=%g",
		c.Name, rep.FirstDisagreeIdx, rep.MaxAbsDiff)
}
