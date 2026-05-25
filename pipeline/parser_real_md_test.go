package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
)

// TestParser_RealBundleMD_StrictMatch is Phase 7 Block C7's end-to-end parser
// differential against Python en_core_web_md. Strict 100% UAS (Head match)
// and DEP-label match on the 8 fixture sentences vs Python_md.
//
// Why strict 100%: same architecture (greedy ArcEager + PrecomputableAffine
// lower / Linear upper) with beam_width=1 — deterministic. The only difference
// vs sm is the upstream tok2vec carries a StaticVectors arm, and that's
// already verified at 1e-6 per-layer parity (C3). Divergence = real port bug.
func TestParser_RealBundleMD_StrictMatch(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_md")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_md not present: %s", bundlePath)
	}
	rawGold, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "parser_arcs_md.json"))
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
	require.Len(t, golden, 8)

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
		require.Equal(t, len(want), d.NumTokens())
		for i, w := range want {
			total++
			gotHead := d.Tokens[i].Head
			gotDep, _ := ss.Lookup(d.Tokens[i].Dep)
			if gotHead == w.Head {
				matchHead++
			} else {
				t.Logf("md HEAD case %s tok %d (%q): want=%d got=%d", c.ID, i, w.Text, w.Head, gotHead)
			}
			if gotDep == w.Dep {
				matchDep++
			} else {
				t.Logf("md DEP  case %s tok %d (%q): want=%q got=%q", c.ID, i, w.Text, w.Dep, gotDep)
			}
		}
	}
	t.Logf("md UAS = %d/%d, LAS-component = %d/%d", matchHead, total, matchDep, total)
	require.Equalf(t, total, matchHead, "md UAS: %d/%d", matchHead, total)
	require.Equalf(t, total, matchDep, "md DEP: %d/%d", matchDep, total)
}
