package layers

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestIntsGetitem_ColumnSlice(t *testing.T) {
	b, err := os.ReadFile(goldenPath(t, "ints_getitem.json"))
	require.NoError(t, err)
	var g struct {
		Dims  map[string]int `json:"dims"`
		Input struct {
			Shape []int    `json:"shape"`
			Data  []uint64 `json:"data"`
		} `json:"input"`
		Output struct {
			Shape []int    `json:"shape"`
			Data  []uint64 `json:"data"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(b, &g))

	ops := gonum.New()
	m := IntsGetitem(ops, g.Dims["col"])

	X := nn.Uint64s2d{Data: g.Input.Data, Rows: g.Dims["N"], Cols: g.Dims["K"]}
	raw, err := m.Predict(X)
	require.NoError(t, err)
	Y := raw.(nn.Uint64s1d)
	require.Equal(t, g.Output.Data, Y.Data)
}
