package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
)

// TestParser_RealBundleLG_StrictMatch is Phase 7 Block C10's end-to-end parser
// differential against Python en_core_web_lg. Strict 100% UAS + DEP-label
// match on the 8 fixture sentences vs Python_lg.
//
// SKIPped when lg is not downloaded (~425 MB).
func TestParser_RealBundleLG_StrictMatch(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_lg")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_lg not present: %s", bundlePath)
	}
	rawGold, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "parser_arcs_lg.json"))
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

	b, err := bundle.FromDisk(bundlePath)
	require.NoError(t, err)
	ss := b.Vocab.StringStore()

	matchHead, matchDep, total := 0, 0, 0
	for _, c := range casesFile.Cases {
		want := golden[c.ID]
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
				t.Logf("lg HEAD case %s tok %d (%q): want=%d got=%d", c.ID, i, w.Text, w.Head, gotHead)
			}
			if gotDep == w.Dep {
				matchDep++
			} else {
				t.Logf("lg DEP  case %s tok %d (%q): want=%q got=%q", c.ID, i, w.Text, w.Dep, gotDep)
			}
		}
	}
	t.Logf("lg UAS = %d/%d, LAS-component = %d/%d", matchHead, total, matchDep, total)
	require.Equalf(t, total, matchHead, "lg UAS: %d/%d", matchHead, total)
	require.Equalf(t, total, matchDep, "lg DEP: %d/%d", matchDep, total)
}
