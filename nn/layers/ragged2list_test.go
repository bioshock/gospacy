package layers

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestRagged2List_Forward(t *testing.T) {
	b, err := os.ReadFile(goldenPath(t, "ragged2list.json"))
	require.NoError(t, err)
	var g struct {
		Cols        int       `json:"cols"`
		Lengths     []int32   `json:"lengths"`
		InputData   []float32 `json:"input_data"`
		OutputItems []struct {
			Shape []int     `json:"shape"`
			Data  []float32 `json:"data"`
		} `json:"output_items"`
	}
	require.NoError(t, json.Unmarshal(b, &g))

	ops := gonum.New()
	m := Ragged2List(ops)

	raw, err := m.Predict(nn.Ragged{Data: g.InputData, Lengths: g.Lengths, Cols: g.Cols})
	require.NoError(t, err)
	out := raw.(nn.FloatList)
	require.Len(t, out.Items, len(g.OutputItems))
	for i, want := range g.OutputItems {
		require.Equal(t, want.Shape[0], out.Items[i].Rows, "item %d Rows", i)
		require.Equal(t, want.Shape[1], out.Items[i].Cols, "item %d Cols", i)
		diff.AssertFloats(t, want.Data, out.Items[i].Data, 0, "ragged2list item")
	}
}

// TestRagged2List_RoundtripWithList2Ragged verifies that List2Ragged followed
// by Ragged2List reconstructs the original FloatList exactly.
func TestRagged2List_RoundtripWithList2Ragged(t *testing.T) {
	ops := gonum.New()
	original := nn.FloatList{Items: []nn.Floats2d{
		{Data: []float32{1, 2, 3, 4}, Rows: 2, Cols: 2},
		{Data: []float32{5, 6}, Rows: 1, Cols: 2},
		{Data: []float32{7, 8, 9, 10, 11, 12}, Rows: 3, Cols: 2},
	}}

	// Forward: FloatList → Ragged
	l2r := List2Ragged(ops)
	raggedRaw, err := l2r.Predict(original)
	require.NoError(t, err)
	ragged := raggedRaw.(nn.Ragged)

	// Inverse: Ragged → FloatList
	r2l := Ragged2List(ops)
	listRaw, err := r2l.Predict(ragged)
	require.NoError(t, err)
	result := listRaw.(nn.FloatList)

	require.Len(t, result.Items, len(original.Items))
	for i, orig := range original.Items {
		require.Equal(t, orig.Rows, result.Items[i].Rows, "item %d Rows", i)
		require.Equal(t, orig.Cols, result.Items[i].Cols, "item %d Cols", i)
		require.Equal(t, orig.Data, result.Items[i].Data, "item %d Data", i)
	}
}
