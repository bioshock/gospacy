package gonum

import (
	"encoding/json"
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func TestGatherAdd_AgainstSample(t *testing.T) {
	fx, err := diff.LoadOpFixture(goldenSamplePath(t))
	require.NoError(t, err)
	c, ok := fx.Ops["gather_add"]
	require.True(t, ok)

	table, _ := c.ArrayInput("table")
	var idx struct {
		Shape []int   `json:"shape"`
		Dtype string  `json:"dtype"`
		Data  []int32 `json:"data"`
	}
	require.NoError(t, json.Unmarshal(c.Inputs["indices"], &idx))
	T, _ := c.IntInput("T")
	w, _ := c.IntInput("w")
	N, _ := c.IntInput("N")
	K, _ := c.IntInput("K")

	out := make([]float32, N*w)
	GatherAdd(out, table.Data, T, w, idx.Data, N, K)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-6, RelMax: 1e-5})
	require.Truef(t, rep.Equal(), "gather_add mismatch (case=%q)", c.Name)
}
