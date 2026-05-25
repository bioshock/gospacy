package layers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

type layerNormGolden struct {
	Dims   map[string]int `json:"dims"`
	Input  goldenArr      `json:"input"`
	G      goldenArr      `json:"G"`
	B      goldenArr      `json:"b"`
	Output goldenArr      `json:"output"`
}

type goldenArr struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

func goldenPath(t *testing.T, name string) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "testdata", "golden", name)
}

func TestLayerNorm_Forward_MatchesThincGolden(t *testing.T) {
	b, err := os.ReadFile(goldenPath(t, "layernorm.json"))
	require.NoError(t, err)
	var g layerNormGolden
	require.NoError(t, json.Unmarshal(b, &g))

	ops := gonum.New()
	N, nI := g.Dims["N"], g.Dims["nI"]
	m := LayerNorm(ops, nI)
	m.Params["G"] = g.G.Data
	m.Params["b"] = g.B.Data

	X := nn.Floats2d{Data: g.Input.Data, Rows: N, Cols: nI}
	raw, err := m.Predict(X)
	require.NoError(t, err)
	Y := raw.(nn.Floats2d)

	require.Equal(t, N, Y.Rows)
	require.Equal(t, nI, Y.Cols)
	diff.AssertFloats(t, g.Output.Data, Y.Data, 1e-5, "LayerNorm forward")
}
