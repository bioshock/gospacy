package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
)

// TestTagger_RealBundleLG_StrictMatch is Phase 7 Block C10's end-to-end
// differential against Python en_core_web_lg. Strict 100% per-token match on
// Tag, POS, Morph, Lemma over the 8 fixture sentences vs Python_lg.
//
// lg shares md's architecture (StaticVectors arm in tok2vec) but ships a
// ~17x larger vector matrix (342918 unique rows vs md's 20000). With 684830
// key2row entries in both, lg simply has less hash-collision lossiness.
// The test confirms that the Go side's populated-vector lookup scales to
// the larger matrix without numerical drift.
//
// SKIPped when lg is not downloaded (~425 MB).
func TestTagger_RealBundleLG_StrictMatch(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_lg")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_lg not present: %s", bundlePath)
	}

	type taggerTok struct {
		Text string `json:"text"`
		Tag  string `json:"tag"`
		POS  string `json:"pos"`
	}
	var taggerGold map[string][]taggerTok
	rawTagger, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "tagger_lg.json"))
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(rawTagger, &taggerGold))

	type attrTok struct {
		Text  string `json:"text"`
		Tag   string `json:"tag"`
		POS   string `json:"pos"`
		Morph string `json:"morph"`
	}
	var attrGold map[string][]attrTok
	rawAttr, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "attribute_ruler_lg.json"))
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(rawAttr, &attrGold))

	type lemmaTok struct {
		Text  string `json:"text"`
		Lemma string `json:"lemma"`
		POS   string `json:"pos"`
	}
	var lemmaGold map[string][]lemmaTok
	rawLemma, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "lemmatizer_lg.json"))
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

	b, err := bundle.FromDisk(bundlePath)
	require.NoError(t, err)
	ss := b.Vocab.StringStore()

	matchTag, matchPOS, matchMorph, matchLemma, total := 0, 0, 0, 0, 0
	for _, c := range casesFile.Cases {
		wantAttr := attrGold[c.ID]
		wantLemma := lemmaGold[c.ID]
		require.NotNilf(t, wantAttr, "attribute_ruler golden missing for %s", c.ID)
		require.NotNilf(t, wantLemma, "lemmatizer golden missing for %s", c.ID)
		d, err := b.Pipe(c.Text)
		require.NoErrorf(t, err, "Pipe(%q)", c.Text)
		require.Equal(t, len(wantAttr), d.NumTokens())
		for i := range wantAttr {
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
				t.Logf("lg case %s tok %d (%q): TAG want=%q got=%q | POS want=%q got=%q | MORPH want=%q got=%q | LEMMA want=%q got=%q",
					c.ID, i, wantAttr[i].Text,
					wantAttr[i].Tag, gotTag,
					wantAttr[i].POS, gotPOS,
					wantAttr[i].Morph, gotMorph,
					wantLemma[i].Lemma, gotLemma)
			}
		}
	}
	t.Logf("lg totals: Tag %d/%d, POS %d/%d, Morph %d/%d, Lemma %d/%d",
		matchTag, total, matchPOS, total, matchMorph, total, matchLemma, total)
	require.Equalf(t, total, matchTag, "lg TAG: %d/%d", matchTag, total)
	require.Equalf(t, total, matchPOS, "lg POS: %d/%d", matchPOS, total)
	require.Equalf(t, total, matchMorph, "lg MORPH: %d/%d", matchMorph, total)
	require.Equalf(t, total, matchLemma, "lg LEMMA: %d/%d", matchLemma, total)
}
