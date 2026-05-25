package layers_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/bioshock/gospacy/v3/nn/layers"
)

type pcaGoldenArr struct {
	Shape []int     `json:"shape"`
	Dtype string    `json:"dtype"`
	Data  []float32 `json:"data"`
}

type pcaGolden struct {
	Description string         `json:"description"`
	Dims        map[string]int `json:"dims"`
	Input       pcaGoldenArr   `json:"input"`
	W           pcaGoldenArr   `json:"W"`
	B           pcaGoldenArr   `json:"b"`
	Pad         pcaGoldenArr   `json:"pad"`
	Output      pcaGoldenArr   `json:"output"`
}

func TestPrecomputableAffine_MatchesPythonGolden(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "precomputable_affine.json"))
	require.NoError(t, err)
	var g pcaGolden
	require.NoError(t, json.Unmarshal(raw, &g))

	nF, nO, nP, nI := g.Dims["nF"], g.Dims["nO"], g.Dims["nP"], g.Dims["nI"]
	T := g.Dims["T"]

	ops := gonum.New()
	m := layers.PrecomputableAffine(ops, nO, nI, nF, nP)
	m.Params["W"] = g.W.Data     // (nF, nO, nP, nI) flat
	m.Params["b"] = g.B.Data     // (nO, nP) flat
	m.Params["pad"] = g.Pad.Data // (1, nF, nO, nP) flat

	X := nn.Floats2d{Data: g.Input.Data, Rows: T, Cols: nI}
	raw2, err := m.Predict(X)
	require.NoError(t, err)
	got, ok := raw2.(nn.Floats2d)
	require.True(t, ok, "expected Floats2d, got %T", raw2)
	require.Equal(t, (T+1)*nF, got.Rows, "rows should be (T+1)*nF")
	require.Equal(t, nO*nP, got.Cols, "cols should be nO*nP")
	require.Len(t, got.Data, len(g.Output.Data))

	for i, want := range g.Output.Data {
		if math.Abs(float64(got.Data[i]-want)) > 1e-4 {
			t.Fatalf("idx=%d: got %f want %f", i, got.Data[i], want)
		}
	}
}
