package registry_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/bioshock/gospacy/v3/lang/en"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/registry"
	"github.com/bioshock/gospacy/v3/tokenizer"
	"github.com/bioshock/gospacy/v3/vocab"
	"github.com/stretchr/testify/require"
)

func goldenPath(t *testing.T, name string) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "testdata", "golden", name)
}

func TestExtractFeatures_Forward_MatchesPython(t *testing.T) {
	raw, err := os.ReadFile(goldenPath(t, "extract_features.json"))
	require.NoError(t, err)
	var g struct {
		Columns []string `json:"columns"`
		Docs    []struct {
			Text  string   `json:"text"`
			Shape []int    `json:"shape"`
			Data  []uint64 `json:"data"`
		} `json:"docs"`
	}
	require.NoError(t, json.Unmarshal(raw, &g))

	cfg := map[string]any{"columns": toAnyList(g.Columns)}
	model, err := registry.Build("spacy.FeatureExtractor.v1", cfg)
	require.NoError(t, err)
	require.NotNil(t, model.Forward, "ExtractFeatures.Forward must be set (Task 6)")

	rules, err := en.MakeRules()
	require.NoError(t, err)
	tk := tokenizer.New(rules)
	v := vocab.NewVocab()

	docs := make([]any, len(g.Docs))
	for i, dg := range g.Docs {
		docs[i] = tk.ToDoc(v, dg.Text)
	}

	rawOut, err := model.Predict(docs)
	require.NoError(t, err)
	items := rawOut.([]nn.Uint64s2d)
	require.Len(t, items, len(g.Docs))
	for i, want := range g.Docs {
		require.Equal(t, want.Shape[0], items[i].Rows, "doc %d Rows", i)
		require.Equal(t, want.Shape[1], items[i].Cols, "doc %d Cols", i)
		require.Equal(t, want.Data, items[i].Data, "doc %d feature matrix", i)
	}
}

func toAnyList(s []string) []any {
	out := make([]any, len(s))
	for i, v := range s {
		out[i] = v
	}
	return out
}
