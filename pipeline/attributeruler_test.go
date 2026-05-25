package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
	"github.com/bioshock/gospacy/v3/pipeline"
)

// arGoldenTok is one row of testdata/golden/attribute_ruler.json — the
// (text, tag, pos, morph) tuple Python emitted after running the
// AttributeRuler pipe.
type arGoldenTok struct {
	Text  string `json:"text"`
	Tag   string `json:"tag"`
	POS   string `json:"pos"`
	Morph string `json:"morph"`
}

// taggerGoldenTok is one row of testdata/golden/tagger.json — used to
// pre-seed Token.Tag/Token.Text before AttributeRuler.Apply runs, since
// the real-bundle Tagger forward is deferred to Phase 4.5 (see plan
// mid-execution amendment). Feeding the golden tagger output into Apply
// isolates the rule-application logic from the tok2vec parity work.
type taggerGoldenTok struct {
	Text string `json:"text"`
	Tag  string `json:"tag"`
	POS  string `json:"pos"`
}

// isKnownARDivergence returns true for (caseID, tokIdx) pairs that are
// known to diverge from Python because the matching pattern depends on
// parser output (DEP), which Phase 4 does not produce. See
// KNOWN_DIVERGENCES.md for the full justification. Limited to the exact
// (case, token) pairs — broader scope would mask real regressions.
//
// Both pairs share the same root cause and one case (s07), matching the
// "Single-case divergence is acceptable" allowance in the plan.
func isKnownARDivergence(caseID string, tokIdx int) bool {
	if caseID != "s07" {
		return false
	}
	// tok 3 "than" → SCONJ requires DEP=mark
	// tok 5 "did"  → VERB  requires DEP∈{ROOT,advcl,...} overriding the
	//                 default AUX from {TAG:VBD, LOWER:'did'}
	return tokIdx == 3 || tokIdx == 5
}

// TestAttributeRuler_DifferentialMorph runs AttributeRuler.Apply against
// Docs whose Tokens have been hand-seeded with the golden tagger output,
// and asserts that the resulting Token.POS / Token.Morph match Python's
// attribute_ruler output exactly across all 8 pipeline_cases.
//
// Why pre-seeded tags rather than tg.Apply(d): the real-bundle Tagger
// forward needs tok2vec layer parity, deferred to Phase 4.5. This test
// proves the rule-application logic against real Python patterns
// independently of that work — patterns key off TAG, so feeding them the
// authoritative Python tags exercises every pattern that the 8 sentences
// trigger.
func TestAttributeRuler_DifferentialMorph(t *testing.T) {
	modelPath := "../testdata/models/en_core_web_sm"
	if _, err := os.Stat(filepath.Join(modelPath, "meta.json")); err != nil {
		t.Skip("model not downloaded")
	}
	b, err := bundle.FromDisk(modelPath)
	require.NoError(t, err)

	ar, err := pipeline.NewAttributeRuler(b)
	require.NoError(t, err)

	rawAR, err := os.ReadFile("../testdata/golden/attribute_ruler.json")
	require.NoError(t, err)
	var goldenAR map[string][]arGoldenTok
	require.NoError(t, json.Unmarshal(rawAR, &goldenAR))

	rawTag, err := os.ReadFile("../testdata/golden/tagger.json")
	require.NoError(t, err)
	var goldenTag map[string][]taggerGoldenTok
	require.NoError(t, json.Unmarshal(rawTag, &goldenTag))

	rawCases, err := os.ReadFile("../testharness/pipeline_cases.json")
	require.NoError(t, err)
	var casesFile struct {
		Cases []struct {
			ID   string `json:"id"`
			Text string `json:"text"`
		} `json:"cases"`
	}
	require.NoError(t, json.Unmarshal(rawCases, &casesFile))

	ss := b.Vocab.StringStore()

	for _, c := range casesFile.Cases {
		t.Run(c.ID, func(t *testing.T) {
			wantAR, ok := goldenAR[c.ID]
			require.Truef(t, ok, "missing attribute_ruler golden for %s", c.ID)
			wantTag, ok := goldenTag[c.ID]
			require.Truef(t, ok, "missing tagger golden for %s", c.ID)
			require.Equalf(t, len(wantAR), len(wantTag),
				"golden tagger/AR token counts differ for %s", c.ID)

			d := b.Tokenizer.ToDoc(b.Vocab, c.Text)
			require.Equalf(t, len(wantAR), d.NumTokens(),
				"tokenizer produced %d tokens but golden has %d for %s",
				d.NumTokens(), len(wantAR), c.ID)

			// Seed Token.Tag and Token.Text from the Python tagger golden.
			// Token.Orth is already set by the tokenizer; we don't touch it.
			for i := range d.Tokens {
				d.Tokens[i].Text = wantTag[i].Text
				d.Tokens[i].Tag = ss.Add(wantTag[i].Tag)
			}

			require.NoError(t, ar.Apply(d))

			for i, want := range wantAR {
				if isKnownARDivergence(c.ID, i) {
					// Documented in KNOWN_DIVERGENCES.md — parser-dependent
					// pattern (DEP=mark) cannot fire in Phase 4. Log so the
					// breach is visible per Rule 12.
					t.Logf("skip known divergence: case %s tok %d %q (see KNOWN_DIVERGENCES.md)",
						c.ID, i, want.Text)
					continue
				}
				gotPOS, _ := ss.Lookup(d.Tokens[i].POS)
				require.Equalf(t, want.POS, gotPOS,
					"case %s tok %d %q (tag=%s): POS want %q got %q",
					c.ID, i, want.Text, want.Tag, want.POS, gotPOS)
				require.Equalf(t, want.Morph, d.Tokens[i].Morph,
					"case %s tok %d %q (tag=%s): Morph want %q got %q",
					c.ID, i, want.Text, want.Tag, want.Morph, d.Tokens[i].Morph)
			}
		})
	}
}

// TestAttributeRuler_RealBundleEndToEnd runs Bundle.Pipe (tokenize → tok2vec →
// tagger → attribute_ruler) on the 8 pipeline_cases and asserts per-token
// POS/Morph agreement with Python's attribute_ruler golden. Phase 4 ran the
// AR test against pre-seeded golden tags; with Phase 4.5 Task 13 wiring the
// real tagger forward, the same assertions now flow through Tagger output.
//
// Threshold: POS and Morph agreement must each be within 2 tokens of perfect.
// That slack covers the 2 documented DEP-dependent divergences on s07
// ("than", "did"); see KNOWN_DIVERGENCES.md. Anything stricter is better and
// the test stays green; the floor here only catches regressions.
func TestAttributeRuler_RealBundleEndToEnd(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present: %s", bundlePath)
	}
	goldenPath := filepath.Join("..", "testdata", "golden", "attribute_ruler.json")
	raw, err := os.ReadFile(goldenPath)
	require.NoError(t, err)
	var golden map[string][]arGoldenTok
	require.NoError(t, json.Unmarshal(raw, &golden))

	rawCases, err := os.ReadFile("../testharness/pipeline_cases.json")
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

	posMatch, morphMatch, total := 0, 0, 0
	for _, c := range casesFile.Cases {
		want, ok := golden[c.ID]
		require.Truef(t, ok, "missing AR golden for %s", c.ID)
		d, err := b.Pipe(c.Text)
		require.NoError(t, err)
		require.Equalf(t, len(want), d.NumTokens(),
			"case %s: pipe produced %d tokens, golden has %d", c.ID, d.NumTokens(), len(want))
		ss := b.Vocab.StringStore()
		for i, w := range want {
			gotPOS, _ := ss.Lookup(d.Tokens[i].POS)
			total++
			if gotPOS == w.POS {
				posMatch++
			} else {
				t.Logf("POS miss: case %s tok %d %q: want %q got %q", c.ID, i, w.Text, w.POS, gotPOS)
			}
			if d.Tokens[i].Morph == w.Morph {
				morphMatch++
			} else {
				t.Logf("Morph miss: case %s tok %d %q: want %q got %q", c.ID, i, w.Text, w.Morph, d.Tokens[i].Morph)
			}
		}
	}
	// Phase 4 documented 2-token POS divergence on s07 ("than", "did") in
	// KNOWN_DIVERGENCES.md (DEP-keyed AR patterns, parser-dependent).
	// With real Tag now flowing through, the divergence count must be
	// AT MOST 2 (the 2 documented). Stricter = better; do not regress.
	require.GreaterOrEqualf(t, posMatch, total-2,
		"AR POS agreement: %d/%d (allowed slack: 2)", posMatch, total)
	require.GreaterOrEqualf(t, morphMatch, total-2,
		"AR Morph agreement: %d/%d (allowed slack: 2)", morphMatch, total)
	t.Logf("AR end-to-end: POS %d/%d, Morph %d/%d", posMatch, total, morphMatch, total)
}
