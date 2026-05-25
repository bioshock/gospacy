package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
)

// TestParser_RealBundle_ProducesPythonMatchingArcs is the Phase 5 end-to-end
// differential. Loads en_core_web_sm, runs Bundle.Pipe on the 8 fixture
// sentences, and asserts Token.Head and Token.Dep exactly match Python.
//
// Why strict 100%: greedy ArcEager with beam_width=1 is deterministic. Any
// per-token divergence reflects either (a) a sum_state_features bug, (b) a
// maxout/scorer rounding bug above 1e-4, or (c) a TransitionSystem ordering
// bug.
func TestParser_RealBundle_ProducesPythonMatchingArcs(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present: %s", bundlePath)
	}
	rawGold, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "parser_arcs.json"))
	require.NoError(t, err)
	type goldenTok struct {
		I    int    `json:"i"`
		Text string `json:"text"`
		Head int    `json:"head"`
		Dep  string `json:"dep"`
		POS  string `json:"pos"`
	}
	var golden map[string][]goldenTok
	require.NoError(t, json.Unmarshal(rawGold, &golden))
	require.Len(t, golden, 8, "parser_arcs.json should have 8 cases")

	rawCases, err := os.ReadFile(filepath.Join("..", "testharness", "pipeline_cases.json"))
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

	matchHead, matchDep, total := 0, 0, 0
	for _, c := range casesFile.Cases {
		want, ok := golden[c.ID]
		require.Truef(t, ok, "missing golden for %s", c.ID)

		d, err := b.Pipe(c.Text)
		require.NoErrorf(t, err, "Pipe(%q)", c.Text)
		require.Equalf(t, len(want), d.NumTokens(),
			"case %s: tokenizer produced %d tokens but golden has %d",
			c.ID, d.NumTokens(), len(want))

		for i, w := range want {
			total++
			gotHead := d.Tokens[i].Head
			gotDep, _ := ss.Lookup(d.Tokens[i].Dep)
			if gotHead == w.Head {
				matchHead++
			} else {
				t.Logf("HEAD case %s tok %d (%q): want=%d got=%d", c.ID, i, w.Text, w.Head, gotHead)
			}
			if gotDep == w.Dep {
				matchDep++
			} else {
				t.Logf("DEP  case %s tok %d (%q): want=%q got=%q", c.ID, i, w.Text, w.Dep, gotDep)
			}
		}
	}
	t.Logf("UAS = %d/%d, LAS-component (dep label) = %d/%d", matchHead, total, matchDep, total)
	require.Equalf(t, total, matchHead, "UAS: %d/%d", matchHead, total)
	require.Equalf(t, total, matchDep, "DEP label: %d/%d", matchDep, total)
}

// TestParser_RealBundle_ClosesS07POSGap asserts the second-order effect: with
// parser DEP set, AttributeRuler reclassifies s07 tokens 3 ("than") and 5
// ("did") from ADP/AUX to SCONJ/VERB, matching Python.
func TestParser_RealBundle_ClosesS07POSGap(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present: %s", bundlePath)
	}
	b, err := bundle.FromDisk(bundlePath)
	require.NoError(t, err)
	d, err := b.Pipe("John ran faster than Mary did.")
	require.NoError(t, err)
	ss := b.Vocab.StringStore()
	require.Equal(t, 7, d.NumTokens())
	tok3POS, _ := ss.Lookup(d.Tokens[3].POS)
	tok5POS, _ := ss.Lookup(d.Tokens[5].POS)
	require.Equal(t, "SCONJ", tok3POS, "tok 3 (than) POS should be SCONJ after parser writes DEP")
	require.Equal(t, "VERB", tok5POS, "tok 5 (did) POS should be VERB after parser writes DEP")
}
