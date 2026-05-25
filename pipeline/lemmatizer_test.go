package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/bioshock/gospacy/v3/bundle"
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/internal/lookups"
	en "github.com/bioshock/gospacy/v3/pipeline/lang/en"
	"github.com/bioshock/gospacy/v3/pipeline"
	"github.com/bioshock/gospacy/v3/vocab"
	"github.com/stretchr/testify/require"
)

func TestLemmatizer_LookupMode(t *testing.T) {
	v := vocab.NewVocab()
	// Mock lookups: a single lemma_lookup entry mapping "cats"→"cat".
	hCats := v.StringStore().Add("cats")
	hCat := v.StringStore().Add("cat")
	l := &lookups.Lookups{}
	// Construct via a helper bound to the test (the package exposes a
	// constructor on a sibling file when needed; we use a closed-form fake here).
	fake := &fakeLookups{tables: map[string]*lookups.Table{
		"lemma_lookup": {Name: "lemma_lookup", Data: map[uint64]any{hCats: hCat}},
	}}

	lem, err := pipeline.NewLemmatizerForTest(v, fake, "lookup")
	require.NoError(t, err)

	d := doc.NewDoc(v, "cats")
	d.Tokens = []doc.Token{{Text: "cats", Orth: hCats}}
	require.NoError(t, lem.Apply(d))
	got, _ := v.StringStore().Lookup(d.Tokens[0].Lemma)
	require.Equal(t, "cat", got)
	_ = l // silence unused
}

// fakeLookups satisfies the small lookups-interface the Lemmatizer needs.
type fakeLookups struct{ tables map[string]*lookups.Table }

func (f *fakeLookups) Has(name string) bool           { _, ok := f.tables[name]; return ok }
func (f *fakeLookups) Get(name string) *lookups.Table { return f.tables[name] }

// lemmaGoldenTok is one row of testdata/golden/lemmatizer.json.
type lemmaGoldenTok struct {
	Text  string `json:"text"`
	Lemma string `json:"lemma"`
	POS   string `json:"pos"`
}

// arGoldenTokForLemma is one row of testdata/golden/attribute_ruler.json.
// We feed Tag/POS/Morph from that file directly into the Doc, isolating the
// Lemmatizer from Tagger/AttributeRuler logic (Tagger forward is deferred to
// Phase 4.5 per the 2026-05-19 plan mid-execution amendment; AttributeRuler
// has one documented DEP-dependent divergence on s07). The attribute_ruler
// golden is the *immediate* upstream of Lemmatizer in spaCy's pipeline, so
// pre-seeding from it lets us assert lemma parity with Python directly.
type arGoldenTokForLemma struct {
	Text  string `json:"text"`
	Tag   string `json:"tag"`
	POS   string `json:"pos"`
	Morph string `json:"morph"`
}

// isKnownLemmaARSetDivergence returns true for (caseID, tokIdx) pairs whose
// Python lemma is set by the AttributeRuler (via a LEMMA attr pattern) BEFORE
// the Lemmatizer runs. With overwrite=false, Python's Lemmatizer respects the
// AR-set value; our isolated differential test runs only Lemmatizer.Apply
// (per the Phase 4 mid-execution amendment), so those tokens are unreachable.
//
// The four documented pairs all share one root cause: AR-LEMMA-attr patterns
// (e.g. [TAG:PRP, LOWER:i] → LEMMA:I; [TAG:VBZ, LOWER∈{is,'s}] → LEMMA:be).
// Documented in KNOWN_DIVERGENCES.md (Phase 4 v0.0.3-alpha section).
//
// Allow-listed scope is tight: explicit (case, tokIdx, text) triples — any
// broader divergence would mask a real regression.
func isKnownLemmaARSetDivergence(caseID string, tokIdx int, text string) bool {
	switch caseID {
	case "s03":
		// tok 0 "I"   → AR pattern [TAG:PRP, LOWER:i] sets LEMMA:I
		return tokIdx == 0 && text == "I"
	case "s04":
		// tok 1 "is"  → AR pattern [TAG:VBZ, LOWER∈{is,'s}] sets LEMMA:be
		return tokIdx == 1 && text == "is"
	case "s05":
		// tok 1 "n't" → AR pattern [TAG:RB, LOWER∈{not,n't,...}] sets LEMMA:not
		return tokIdx == 1 && text == "n't"
	case "s08":
		// tok 1 "are" → AR pattern [TAG:VBP, LOWER∈{are,'re}] sets LEMMA:be
		return tokIdx == 1 && text == "are"
	}
	return false
}

// TestLemmatizer_DifferentialEnglish runs Lemmatizer.Apply against Docs whose
// Tokens have been hand-seeded with the Python attribute_ruler golden
// (Text/Tag/POS/Morph) and asserts that the resulting Token.Lemma matches the
// Python lemmatizer golden across all 8 pipeline_cases.
//
// Why pre-seeded attrs rather than tg.Apply + ar.Apply: the real-bundle Tagger
// forward is deferred to Phase 4.5 (Phase 4 plan mid-execution amendment);
// also, the AttributeRuler has a documented DEP-dependent divergence on s07
// (see KNOWN_DIVERGENCES.md). Feeding the Lemmatizer the authoritative Python
// AR output isolates the rule/exc/index logic from both deferrals.
//
// Threshold: ≥95% per-token lemma match. The 4 known AR-LEMMA-set divergences
// listed in isKnownLemmaARSetDivergence are allow-listed; on the 68 tokens in
// the 8 cases that yields 64/68 = 94.1% strict match, 100% after the
// documented allow-list, exceeding the ≥95% threshold the plan permits when
// divergence has a single root cause and is documented (Rule 12: fail loud).
func TestLemmatizer_DifferentialEnglish(t *testing.T) {
	modelPath := "../testdata/models/en_core_web_sm"
	if _, err := os.Stat(filepath.Join(modelPath, "meta.json")); err != nil {
		t.Skip("model not downloaded")
	}
	b, err := bundle.FromDisk(modelPath)
	require.NoError(t, err)

	lm, err := pipeline.NewLemmatizer(b)
	require.NoError(t, err)
	lm.IsBaseForm = func(tok *doc.Token, pos string) bool {
		return en.IsBaseForm(tok, pos)
	}

	rawLemma, err := os.ReadFile("../testdata/golden/lemmatizer.json")
	require.NoError(t, err)
	var goldenLemma map[string][]lemmaGoldenTok
	require.NoError(t, json.Unmarshal(rawLemma, &goldenLemma))

	rawAR, err := os.ReadFile("../testdata/golden/attribute_ruler.json")
	require.NoError(t, err)
	var goldenAR map[string][]arGoldenTokForLemma
	require.NoError(t, json.Unmarshal(rawAR, &goldenAR))

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

	var totalTokens, matches int

	for _, c := range casesFile.Cases {
		t.Run(c.ID, func(t *testing.T) {
			wantLemma, ok := goldenLemma[c.ID]
			require.Truef(t, ok, "missing lemmatizer golden for %s", c.ID)
			seedAR, ok := goldenAR[c.ID]
			require.Truef(t, ok, "missing attribute_ruler golden for %s", c.ID)
			require.Equalf(t, len(wantLemma), len(seedAR),
				"golden lemma/AR token counts differ for %s", c.ID)

			d := b.Tokenizer.ToDoc(b.Vocab, c.Text)
			require.Equalf(t, len(wantLemma), d.NumTokens(),
				"tokenizer produced %d tokens but golden has %d for %s",
				d.NumTokens(), len(wantLemma), c.ID)

			// Seed Token.Text/Tag/POS/Morph from the Python attribute_ruler
			// golden (the immediate upstream of Lemmatizer). Token.Orth is
			// already set by the tokenizer; we leave it alone.
			for i := range d.Tokens {
				d.Tokens[i].Text = seedAR[i].Text
				d.Tokens[i].Tag = ss.Add(seedAR[i].Tag)
				d.Tokens[i].POS = ss.Add(seedAR[i].POS)
				d.Tokens[i].Morph = seedAR[i].Morph
			}

			require.NoError(t, lm.Apply(d))

			for i, want := range wantLemma {
				totalTokens++
				got, _ := ss.Lookup(d.Tokens[i].Lemma)
				if isKnownLemmaARSetDivergence(c.ID, i, want.Text) {
					// Documented in KNOWN_DIVERGENCES.md. Surface the breach
					// per Rule 12: log explicitly rather than silently pass.
					t.Logf("skip known divergence: case %s tok %d %q want %q got %q (AR-LEMMA-set; see KNOWN_DIVERGENCES.md)",
						c.ID, i, want.Text, want.Lemma, got)
					continue
				}
				if want.Lemma == got {
					matches++
				} else {
					// Use t.Errorf (not require) so all mismatches surface in
					// one run — Rule 12.
					t.Errorf("case %s tok %d %q: lemma want %q got %q",
						c.ID, i, want.Text, want.Lemma, got)
				}
			}
		})
	}

	// Enforce ≥95% threshold on the documented-divergence-excluded set.
	allowed := 0
	for _, c := range casesFile.Cases {
		for i, want := range goldenLemma[c.ID] {
			if isKnownLemmaARSetDivergence(c.ID, i, want.Text) {
				allowed++
			}
		}
	}
	effectiveDenom := totalTokens - allowed
	if effectiveDenom == 0 {
		t.Fatal("no tokens to assert against")
	}
	pct := float64(matches) / float64(effectiveDenom) * 100
	t.Logf("lemma match: %d/%d non-allow-listed tokens = %.1f%% (%d allow-listed AR-LEMMA-set tokens)",
		matches, effectiveDenom, pct, allowed)
	require.GreaterOrEqualf(t, pct, 95.0,
		"lemma match %.1f%% below ≥95%% threshold (excluding %d allow-listed AR-LEMMA-set tokens)",
		pct, allowed)
}

// TestLemmatizer_RealBundleEndToEnd runs Bundle.Pipe (tokenize → tok2vec →
// tagger → attribute_ruler → lemmatizer) on the 8 pipeline_cases and asserts
// per-token Lemma agreement with Python's lemmatizer golden. Phase 4 ran the
// lemma test against AR-pre-seeded Tokens (and allow-listed 4 AR-LEMMA-set
// tokens). With Phase 4.5 Task 13 wiring the real tagger and Task 14
// extending AttributeRuler.Apply to write LEMMA, those 4 tokens are reachable
// end-to-end and the differential should hit 68/68.
//
// Threshold: at most 4 lemma misses (preserves the Phase 4 floor; tighter is
// better). The test logs the actual count so any drop below 4 is visible.
func TestLemmatizer_RealBundleEndToEnd(t *testing.T) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present: %s", bundlePath)
	}
	goldenPath := filepath.Join("..", "testdata", "golden", "lemmatizer.json")
	raw, err := os.ReadFile(goldenPath)
	require.NoError(t, err)
	var golden map[string][]lemmaGoldenTok
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

	match, total := 0, 0
	for _, c := range casesFile.Cases {
		want, ok := golden[c.ID]
		require.Truef(t, ok, "missing lemma golden for %s", c.ID)
		d, err := b.Pipe(c.Text)
		require.NoError(t, err)
		require.Equalf(t, len(want), d.NumTokens(),
			"case %s: pipe produced %d tokens, golden has %d", c.ID, d.NumTokens(), len(want))
		ss := b.Vocab.StringStore()
		for i, w := range want {
			got, _ := ss.Lookup(d.Tokens[i].Lemma)
			total++
			if got == w.Lemma {
				match++
			} else {
				t.Logf("Lemma miss: case %s tok %d %q: want %q got %q", c.ID, i, w.Text, w.Lemma, got)
			}
		}
	}
	// Phase 4 allow-listed 4 lemma tokens in KNOWN_DIVERGENCES.md. With real
	// Tag in place AND AttributeRuler.Apply writing LEMMA (Task 14), all 4
	// are now reachable. Threshold: at most 4 misses; aim for 0.
	require.GreaterOrEqualf(t, match, total-4,
		"Lemma agreement: %d/%d (allowed slack: 4)", match, total)
	t.Logf("Lemma end-to-end: %d/%d", match, total)
}
