package workflows_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
	gospacydoc "github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/pipeline/workflows"
)

// TestWorkflows_RealBundle runs every workflow registered in
// workflows.AllWorkflows() against en_core_web_sm on the 8 fixture sentences
// in testharness/pipeline_cases.json and asserts strict equality with the
// matching Python golden under testdata/golden/workflows/.
//
// Why strict 100%: workflows compose v0.1 fields (POS/Tag/Morph/Lemma/Dep/
// Head/SentStart) that already differential at 100% per token. Any divergence
// here therefore reflects helper drift (Sents/Children/NounChunks/IsStop) or
// a real port gap, not tagger/parser noise.
func TestWorkflows_RealBundle(t *testing.T) {
	bundlePath := filepath.Join("..", "..", "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present: %s", bundlePath)
	}
	wfs := workflows.AllWorkflows()
	if len(wfs) == 0 {
		t.Skip("no workflows registered yet")
	}

	rawCases, err := os.ReadFile(filepath.Join("..", "..", "testharness", "pipeline_cases.json"))
	require.NoError(t, err)
	var casesFile struct {
		Cases []struct {
			ID   string `json:"id"`
			Text string `json:"text"`
		} `json:"cases"`
	}
	require.NoError(t, json.Unmarshal(rawCases, &casesFile))
	require.Len(t, casesFile.Cases, 8)

	b, err := bundle.FromDisk(bundlePath)
	require.NoError(t, err)
	ss := b.Vocab.StringStore()

	// Pipe once per case, reuse across workflows.
	docs := make(map[string]*gospacydoc.Doc, len(casesFile.Cases))
	for _, c := range casesFile.Cases {
		d, err := b.Pipe(c.Text)
		require.NoErrorf(t, err, "Pipe(%q)", c.Text)
		docs[c.ID] = d
	}

	for _, wf := range wfs {
		wf := wf
		t.Run(wf.Name, func(t *testing.T) {
			goldenRaw, err := os.ReadFile(wf.GoldenPath)
			require.NoErrorf(t, err, "read golden %s", wf.GoldenPath)
			var golden map[string]any
			require.NoError(t, json.Unmarshal(goldenRaw, &golden))

			got := map[string]any{}
			for _, c := range casesFile.Cases {
				got[c.ID] = wf.Run(docs[c.ID], ss)
			}

			require.Equalf(t, canonical(t, golden), canonical(t, got),
				"workflow %s diverged", wf.Name)
		})
	}
}

// canonical marshals v through json.Marshal (which sorts Go map keys), then
// re-parses + re-marshals so the result is order-independent.
func canonical(t *testing.T, v any) string {
	t.Helper()
	buf, err := json.Marshal(v)
	require.NoError(t, err)
	var anyV any
	require.NoError(t, json.Unmarshal(buf, &anyV))
	buf2, err := json.Marshal(anyV)
	require.NoError(t, err)
	return string(buf2)
}
