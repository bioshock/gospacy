package gonum

import (
	"encoding/json"
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func TestPad_AgainstSample(t *testing.T) {
	fx, _ := diff.LoadOpFixture(goldenSamplePath(t))
	c := fx.Ops["pad"]
	X, _ := c.ArrayInput("X")
	lengths := loadLengths(t, c)
	T, _ := c.IntInput("T")
	w, _ := c.IntInput("w")
	B, _ := c.IntInput("B")
	maxLen, _ := c.IntInput("max_len")

	out := make([]float32, B*maxLen*w)
	Pad(out, X.Data, T, w, lengths, maxLen)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-7, RelMax: 1e-7})
	require.True(t, rep.Equal())
}

func mustLoadInt32(t *testing.T, c diff.OpCase, key string) []int32 {
	t.Helper()
	var v struct {
		Shape []int   `json:"shape"`
		Dtype string  `json:"dtype"`
		Data  []int32 `json:"data"`
	}
	require.NoError(t, json.Unmarshal(c.Inputs[key], &v))
	return v.Data
}

func TestList2Padded_AgainstSample(t *testing.T) {
	fx, _ := diff.LoadOpFixture(goldenSamplePath(t))
	c := fx.Ops["list2padded"]
	X, _ := c.ArrayInput("X")
	lengths := loadLengths(t, c)
	T, _ := c.IntInput("T")
	w, _ := c.IntInput("w")
	B, _ := c.IntInput("B")
	maxLen, _ := c.IntInput("max_len")

	out := make([]float32, maxLen*B*w)
	sizeAtT := make([]int32, maxLen)
	sortedLengths := make([]int32, B)
	indices := make([]int32, B)

	List2Padded(out, sizeAtT, sortedLengths, indices, X.Data, T, w, lengths, maxLen)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-7, RelMax: 1e-7})
	require.Truef(t, rep.Equal(), "list2padded data mismatch (case=%q)", c.Name)
}

func TestPadded2List_AgainstSample(t *testing.T) {
	fx, _ := diff.LoadOpFixture(goldenSamplePath(t))
	c := fx.Ops["padded2list"]
	paddedData, _ := c.ArrayInput("padded_data")
	sizeAtT := mustLoadInt32(t, c, "size_at_t")
	sortedLengths := mustLoadInt32(t, c, "sorted_lengths")
	indices := mustLoadInt32(t, c, "indices")
	B, _ := c.IntInput("B")
	T, _ := c.IntInput("T")
	w, _ := c.IntInput("w")
	outLengths := mustLoadInt32(t, c, "out_lengths")

	totalRows := 0
	for _, l := range outLengths {
		totalRows += int(l)
	}
	out := make([]float32, totalRows*w)

	Padded2List(out, paddedData.Data, sizeAtT, sortedLengths, indices, B, T, w, outLengths)

	exp, _ := c.Float32Output()
	rep := diff.CompareFloats(exp.Data, out, diff.Tolerance{AbsMax: 1e-7, RelMax: 1e-7})
	require.Truef(t, rep.Equal(), "padded2list mismatch (case=%q)", c.Name)
}
