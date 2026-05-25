package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
)

// TestNER_RealBundleMD_StrictMatch is Phase 7 Block D's end-to-end NER
// differential against Python en_core_web_md. Strict 100% per-token
// EntIOB+EntType match on the 8 fixture sentences.
//
// Why md instead of sm: per the orchestrator override, md is the
// reference bundle for NER (matches the md tagger/parser differentials
// already shipped in Block C). The sm bundle's vector-free tok2vec
// produces noisier scores; md is the cleaner baseline.
//
// Why strict 100%: same scorer shape as the parser (PrecomputableAffine
// lower + Linear upper) with beam_width=1 — deterministic, same inputs
// produce identical outputs. Divergence = real port bug.
//
// Only 3 of the 8 fixtures carry entities (s04: Apple/U.K./$1 billion,
// s05: U.S.A./today, s07: John/Mary). The other 5 are entity-free
// baselines — the strict match still verifies every token reads as O.
func TestNER_RealBundleMD_StrictMatch(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_md")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_md not present: %s", bundlePath)
	}
	rawGold, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "entities_cases_md.json"))
	require.NoError(t, err)
	type goldenTok struct {
		Text    string `json:"text"`
		EntIOB  string `json:"ent_iob"`
		EntType string `json:"ent_type"`
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

	// Map gospacy's internal EntIOB code → spaCy's IOB letter. Internal
	// 0 = missing, 1 = I, 2 = O, 3 = B; spaCy's ent_iob_ exposes
	// "" / "I" / "O" / "B" at inference. (L is collapsed to I — see
	// BiluoApply.LAST sets entIOB=1.)
	iobLetter := func(code uint8) string {
		switch code {
		case 0:
			return ""
		case 1:
			return "I"
		case 2:
			return "O"
		case 3:
			return "B"
		}
		return "?"
	}

	matchIOB, matchType, total := 0, 0, 0
	for _, c := range casesFile.Cases {
		want, ok := golden[c.ID]
		require.Truef(t, ok, "missing golden for %s", c.ID)
		d, err := b.Pipe(c.Text)
		require.NoErrorf(t, err, "Pipe(%q)", c.Text)
		require.Equal(t, len(want), d.NumTokens(),
			"case %s: tokenizer produced %d tokens, golden %d", c.ID, d.NumTokens(), len(want))
		for i, w := range want {
			total++
			gotIOB := iobLetter(d.Tokens[i].EntIOB)
			var gotType string
			if d.Tokens[i].EntType != 0 {
				gotType, _ = ss.Lookup(d.Tokens[i].EntType)
			}
			if gotIOB == w.EntIOB {
				matchIOB++
			} else {
				t.Logf("md IOB  case %s tok %d (%q): want=%q got=%q", c.ID, i, w.Text, w.EntIOB, gotIOB)
			}
			if gotType == w.EntType {
				matchType++
			} else {
				t.Logf("md TYPE case %s tok %d (%q): want=%q got=%q", c.ID, i, w.Text, w.EntType, gotType)
			}
		}
	}
	t.Logf("md NER IOB = %d/%d, Type = %d/%d", matchIOB, total, matchType, total)
	require.Equalf(t, total, matchIOB, "md NER IOB: %d/%d", matchIOB, total)
	require.Equalf(t, total, matchType, "md NER Type: %d/%d", matchType, total)
}
