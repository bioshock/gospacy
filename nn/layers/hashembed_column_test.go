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

func TestHashEmbed_WithColumnAttr_LeafForwardMatchesPython(t *testing.T) {
	b, err := os.ReadFile(goldenPath(t, "hashembed_column.json"))
	require.NoError(t, err)
	var g struct {
		Dims  map[string]int `json:"dims"`
		Attrs struct {
			Column      int     `json:"column"`
			Seed        uint32  `json:"seed"`
			DropoutRate float32 `json:"dropout_rate"`
		} `json:"attrs"`
		E struct {
			Shape []int     `json:"shape"`
			Data  []float32 `json:"data"`
		} `json:"E"`
		Ids    []uint64 `json:"ids"`
		Output struct {
			Shape []int     `json:"shape"`
			Data  []float32 `json:"data"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(b, &g))

	ops := gonum.New()
	m := HashEmbed(ops, g.Dims["nO"], g.Dims["nV"], g.Attrs.Seed)
	m.Params["E"] = g.E.Data
	m.Attrs["column"] = g.Attrs.Column
	m.Attrs["dropout_rate"] = g.Attrs.DropoutRate

	raw, err := m.Predict(nn.Uint64s1d{Data: g.Ids})
	require.NoError(t, err)
	Y := raw.(nn.Floats2d)
	require.Equal(t, g.Output.Shape[0], Y.Rows)
	require.Equal(t, g.Output.Shape[1], Y.Cols)
	diff.AssertFloats(t, g.Output.Data, Y.Data, 1e-5, "HashEmbed (column attr) leaf forward")

	// Attrs round-trip.
	require.Equal(t, g.Attrs.Column, m.Attrs["column"].(int))
	require.Equal(t, g.Attrs.Seed, m.Attrs["seed"].(uint32))
	require.Equal(t, g.Attrs.DropoutRate, m.Attrs["dropout_rate"].(float32))
}
