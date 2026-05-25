package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
)

// TestTagger_RealBundle_ProducesPythonMatchingTags is the Phase 4.5 task-13
// differential: load the real en_core_web_sm bundle and run Bundle.Pipe on
// the 8 fixture sentences from testharness/pipeline_cases.json, asserting
// that every Token.Tag exactly matches Python's tagger output captured in
// testdata/golden/tagger.json.
//
// Why this is the right gate for Phase 4.5: tasks 1–12 brought the tok2vec
// per-layer forward within 1e-4 of Python (verified on s01); the tagger is a
// small softmax on top of that. The differential proves the parity chain is
// end-to-end correct: tokenizer → tok2vec listener → tagger softmax →
// argmax → StringStore.Add → Token.Tag.
//
// Threshold: strict 100% on all 68 tokens across the 8 cases. An earlier
// carve-out for s05 tok 0 ("Do") tracked an upstream tokenizer-exceptions
// gap — the NORM override missing on the "n't" clitic; that gap is closed
// (see internal/cmd/genexceptions/main.go and KNOWN_DIVERGENCES.md history).
func TestTagger_RealBundle_ProducesPythonMatchingTags(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present: %s", bundlePath)
	}

	// Golden tags: map[caseID] -> []{text, tag, pos}.
	rawGold, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "tagger.json"))
	require.NoError(t, err)
	var golden map[string][]struct {
		Text string `json:"text"`
		Tag  string `json:"tag"`
		POS  string `json:"pos"`
	}
	require.NoError(t, json.Unmarshal(rawGold, &golden))
	require.Len(t, golden, 8, "tagger.json should have 8 cases")

	// Source texts: pipeline_cases.json owns the (id, text) pairs.
	rawCases, err := os.ReadFile(filepath.Join("..", "testharness", "pipeline_cases.json"))
	require.NoError(t, err)
	var casesFile struct {
		Cases []struct {
			ID   string `json:"id"`
			Text string `json:"text"`
		} `json:"cases"`
	}
	require.NoError(t, json.Unmarshal(rawCases, &casesFile))
	require.Len(t, casesFile.Cases, 8, "pipeline_cases.json should have 8 cases")

	b, err := bundle.FromDisk(bundlePath)
	require.NoError(t, err)
	ss := b.Vocab.StringStore()

	matchCount, total := 0, 0
	for _, c := range casesFile.Cases {
		wantTokens, ok := golden[c.ID]
		require.Truef(t, ok, "missing tagger golden for %s", c.ID)

		d, err := b.Pipe(c.Text)
		require.NoErrorf(t, err, "Pipe(%q)", c.Text)
		require.Equalf(t, len(wantTokens), d.NumTokens(),
			"case %s: tokenizer produced %d tokens but golden has %d",
			c.ID, d.NumTokens(), len(wantTokens))

		for i, want := range wantTokens {
			gotTag, _ := ss.Lookup(d.Tokens[i].Tag)
			total++
			if gotTag == want.Tag {
				matchCount++
			} else {
				t.Logf("case %s tok %d (%q): want=%q got=%q",
					c.ID, i, want.Text, want.Tag, gotTag)
			}
		}
	}
	// Strict 100% across all 68 tokens. Any miss is a real regression.
	t.Logf("Tag agreement: %d/%d tokens across %d cases",
		matchCount, total, len(casesFile.Cases))
	require.Equalf(t, total, matchCount,
		"Tag agreement: %d/%d — see t.Logf output above for mismatches",
		matchCount, total)
}
