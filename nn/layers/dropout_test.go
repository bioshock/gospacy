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

func TestDropout_Inference_IsIdentity(t *testing.T) {
	b, err := os.ReadFile(goldenPath(t, "dropout.json"))
	require.NoError(t, err)
	var g struct {
		Rate      float64   `json:"rate"`
		IsEnabled bool      `json:"is_enabled"`
		Input     goldenArr `json:"input"`
		Output    goldenArr `json:"output"`
	}
	require.NoError(t, json.Unmarshal(b, &g))

	ops := gonum.New()
	m := Dropout(ops, float32(g.Rate))
	m.Attrs["is_enabled"] = g.IsEnabled

	X := nn.Floats2d{Data: g.Input.Data, Rows: g.Input.Shape[0], Cols: g.Input.Shape[1]}
	raw, err := m.Predict(X)
	require.NoError(t, err)
	Y := raw.(nn.Floats2d)
	require.Equal(t, X.Rows, Y.Rows)
	require.Equal(t, X.Cols, Y.Cols)
	diff.AssertFloats(t, g.Output.Data, Y.Data, 0, "Dropout inference identity")

	// Attrs round-trip.
	require.Equal(t, float32(g.Rate), m.Attrs["dropout_rate"].(float32))
	require.Equal(t, g.IsEnabled, m.Attrs["is_enabled"].(bool))
}
