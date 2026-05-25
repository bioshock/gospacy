package gonum

import (
	"encoding/json"
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func loadLengths(t *testing.T, c diff.OpCase) []int32 {
	t.Helper()
	var v struct {
		Shape []int   `json:"shape"`
		Dtype string  `json:"dtype"`
		Data  []int32 `json:"data"`
	}
	require.NoError(t, json.Unmarshal(c.Inputs["lengths"], &v))
	return v.Data
}

func TestReduceFirst_AgainstSample(t *testing.T) {
	fx, _ := diff.LoadOpFixture(goldenSamplePath(t))
	c := fx.Ops["reduce_first"]
	X, _ := c.ArrayInput("X")
	lengths := loadLengths(t, c)
	T, _ := c.IntInput("T")
	w, _ := c.IntInput("w")
	B, _ := c.IntInput("B")

	out := make([]float32, B*w)
	ReduceFirst(out, X.Data, T, w, lengths)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-7, RelMax: 1e-7})
	require.True(t, rep.Equal())
}

func TestReduceLast_AgainstSample(t *testing.T) {
	fx, _ := diff.LoadOpFixture(goldenSamplePath(t))
	c := fx.Ops["reduce_last"]
	X, _ := c.ArrayInput("X")
	lengths := loadLengths(t, c)
	T, _ := c.IntInput("T")
	w, _ := c.IntInput("w")
	B, _ := c.IntInput("B")

	out := make([]float32, B*w)
	ReduceLast(out, X.Data, T, w, lengths)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-7, RelMax: 1e-7})
	require.True(t, rep.Equal())
}
