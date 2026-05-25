package gonum

import (
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func TestSeq2Col_AgainstSample(t *testing.T) {
	fx, err := diff.LoadOpFixture(goldenSamplePath(t))
	require.NoError(t, err)
	c, ok := fx.Ops["seq2col"]
	require.True(t, ok)

	X, _ := c.ArrayInput("X")
	n, _ := c.IntInput("n")
	w, _ := c.IntInput("w")
	nW, _ := c.IntInput("nW")

	outCols := (2*nW + 1) * w
	out := make([]float32, n*outCols)
	Seq2Col(out, X.Data, n, w, nW)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-6, RelMax: 1e-6})
	require.Truef(t, rep.Equal(),
		"seq2col mismatch (case=%q): first disagree at %d, maxAbsDiff=%g",
		c.Name, rep.FirstDisagreeIdx, rep.MaxAbsDiff)
}
