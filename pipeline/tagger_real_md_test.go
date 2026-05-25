package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
)

// TestTagger_RealBundleMD_StrictMatch is Phase 7 Block C6's end-to-end
// differential against Python en_core_web_md. Loads the md bundle, runs
// Bundle.Pipe on the 8 fixture sentences, asserts strict 100% per-token
// match on Tag, POS, Morph (via AttributeRuler), and Lemma vs Python_md.
//
// Why strict 100%: the md tagger uses the same Softmax-on-Tok2Vec
// architecture as sm — the only architectural difference is the tok2vec
// embed has a 7th StaticVectors arm. The per-layer parity (C3) already pins
// numerical agreement of that arm at 1e-6; downstream the softmax + argmax
// + StringStore.Lookup are deterministic. Any divergence = real port bug.
//
// The 4 attribute slices Tag / POS / Morph / Lemma are checked in one test
// to keep the diagnostic context tight when a single fixture token diverges
// — the test logs `case sX tok i (text): TAG want=X got=Y | POS want=X got=Y
// ...` for any miss, then fails with the aggregate counts.
//
// Goldens: testdata/golden/tagger_md.json (text, tag, pos),
//
//	testdata/golden/attribute_ruler_md.json (morph),
//	testdata/golden/lemmatizer_md.json (lemma).
//
// Regenerated via `GOSPACY_MODEL=en_core_web_md testharness/.venv/bin/python
// testharness/dump_tagger.py` (and dump_attribute_ruler.py, dump_lemmatizer.py).
//
// SKIPped when md is not downloaded.
func TestTagger_RealBundleMD_StrictMatch(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_md")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_md not present: %s", bundlePath)
	}

	type taggerTok struct {
		Text string `json:"text"`
		Tag  string `json:"tag"`
		POS  string `json:"pos"`
	}
	var taggerGold map[string][]taggerTok
	rawTagger, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "tagger_md.json"))
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(rawTagger, &taggerGold))

	type attrTok struct {
		Text  string `json:"text"`
		Tag   string `json:"tag"`
		POS   string `json:"pos"`
		Morph string `json:"morph"`
	}
	var attrGold map[string][]attrTok
	rawAttr, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "attribute_ruler_md.json"))
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(rawAttr, &attrGold))

	type lemmaTok struct {
		Text  string `json:"text"`
		Lemma string `json:"lemma"`
		POS   string `json:"pos"`
	}
	var lemmaGold map[string][]lemmaTok
	rawLemma, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "lemmatizer_md.json"))
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(rawLemma, &lemmaGold))

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

	matchTag, matchPOS, matchMorph, matchLemma, total := 0, 0, 0, 0, 0
	for _, c := range casesFile.Cases {
		wantTagger := taggerGold[c.ID]
		wantAttr := attrGold[c.ID]
		wantLemma := lemmaGold[c.ID]
		require.NotNilf(t, wantTagger, "tagger golden missing for %s", c.ID)
		require.NotNilf(t, wantAttr, "attribute_ruler golden missing for %s", c.ID)
		require.NotNilf(t, wantLemma, "lemmatizer golden missing for %s", c.ID)

		d, err := b.Pipe(c.Text)
		require.NoErrorf(t, err, "Pipe(%q)", c.Text)
		require.Equalf(t, len(wantTagger), d.NumTokens(),
			"case %s: %d tokens, golden has %d", c.ID, d.NumTokens(), len(wantTagger))
		require.Equal(t, len(wantAttr), d.NumTokens())
		require.Equal(t, len(wantLemma), d.NumTokens())

		for i := range wantTagger {
			total++
			gotTag, _ := ss.Lookup(d.Tokens[i].Tag)
			gotPOS, _ := ss.Lookup(d.Tokens[i].POS)
			gotMorph := d.Tokens[i].Morph
			gotLemma, _ := ss.Lookup(d.Tokens[i].Lemma)
			tagOK := gotTag == wantAttr[i].Tag
			posOK := gotPOS == wantAttr[i].POS
			morphOK := gotMorph == wantAttr[i].Morph
			lemmaOK := gotLemma == wantLemma[i].Lemma
			if tagOK {
				matchTag++
			}
			if posOK {
				matchPOS++
			}
			if morphOK {
				matchMorph++
			}
			if lemmaOK {
				matchLemma++
			}
			if !(tagOK && posOK && morphOK && lemmaOK) {
				t.Logf("case %s tok %d (%q): TAG want=%q got=%q | POS want=%q got=%q | MORPH want=%q got=%q | LEMMA want=%q got=%q",
					c.ID, i, wantTagger[i].Text,
					wantAttr[i].Tag, gotTag,
					wantAttr[i].POS, gotPOS,
					wantAttr[i].Morph, gotMorph,
					wantLemma[i].Lemma, gotLemma)
			}
		}
	}
	t.Logf("md totals: Tag %d/%d, POS %d/%d, Morph %d/%d, Lemma %d/%d",
		matchTag, total, matchPOS, total, matchMorph, total, matchLemma, total)
	require.Equalf(t, total, matchTag, "md TAG: %d/%d", matchTag, total)
	require.Equalf(t, total, matchPOS, "md POS: %d/%d", matchPOS, total)
	require.Equalf(t, total, matchMorph, "md MORPH: %d/%d", matchMorph, total)
	require.Equalf(t, total, matchLemma, "md LEMMA: %d/%d", matchLemma, total)
}
