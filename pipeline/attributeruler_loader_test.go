package pipeline

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

func TestAttributeRuler_LoadPatterns(t *testing.T) {
	path := "../testdata/models/en_core_web_sm/attribute_ruler/patterns"
	if _, err := os.Stat(path); err != nil {
		t.Skip("model not downloaded")
	}
	pats, err := loadAttributeRulerPatterns(path)
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(pats), 100, "en_core_web_sm has ~179 patterns")
	// The first pattern is _SP → POS=SPACE.
	require.Equal(t, "_SP", pats[0].TokenSpecs[0].Tag)
	require.Equal(t, "SPACE", pats[0].Attrs.POS)
}

func TestAttributeRuler_PatternFile_Exists(t *testing.T) {
	path := filepath.Join("..", "testdata", "models", "en_core_web_sm",
		"attribute_ruler", "patterns")
	if _, err := os.Stat(path); err != nil {
		t.Skip("en_core_web_sm not present; run testharness/download_assets.sh")
	}
	require.FileExists(t, path)
}

// TestLoaderRecognisesDEP — pattern with DEP=str loads with Dep set and
// Unsupported=false.
func TestLoaderRecognisesDEP(t *testing.T) {
	patBytes, err := msgpack.Marshal([]map[string]any{{
		"patterns": []any{[]any{map[string]any{"TAG": "IN", "DEP": "mark"}}},
		"attrs":    map[string]any{"POS": "SCONJ"},
		"index":    int64(0),
	}})
	require.NoError(t, err)
	tmp := t.TempDir()
	path := filepath.Join(tmp, "patterns")
	require.NoError(t, os.WriteFile(path, patBytes, 0o644))

	pats, err := loadAttributeRulerPatterns(path)
	require.NoError(t, err)
	require.Len(t, pats, 1)
	require.False(t, pats[0].Unsupported)
	require.Equal(t, "mark", pats[0].TokenSpecs[0].Dep)
}

// TestMatcher_DepAndTagNotIn_Match exercises the Phase 5 Dep / TagNotIn
// extensions on matchPattern directly. Constructs hand-built specs and
// confirms (1) Dep equality match fires, (2) TagNotIn blocks a token whose
// Tag is in the negated set, (3) DepIn fires on set membership.
func TestMatcher_DepAndTagNotIn_Match(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	tagIN := ss.Add("IN")
	depMark := ss.Add("mark")
	depAdvmod := ss.Add("advmod")

	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "than", Tag: tagIN, Dep: depMark},
		{Text: "early", Tag: ss.Add("RB"), Dep: depAdvmod},
	}}

	// Pattern 1: Tag=IN AND Dep=mark — should match tok 0.
	p1 := arPattern{TokenSpecs: []arTokenSpec{{Tag: "IN", Dep: "mark"}}}
	require.Equal(t, []int{0}, matchPattern(d, p1))

	// Pattern 2: TagNotIn=[IN, ""] — blocks tok 0 (Tag==IN). Tok 1 has Tag=RB,
	// not in {IN, ""}, but the "" sentinel checks Tag != 0; tok 1's Tag is
	// set so the sentinel does not block. Expect tok 1 matches.
	p2 := arPattern{TokenSpecs: []arTokenSpec{{TagNotIn: []string{"IN", ""}}}}
	require.Equal(t, []int{1}, matchPattern(d, p2))

	// Pattern 3: DepIn=[mark, advmod] — matches both tokens.
	p3 := arPattern{TokenSpecs: []arTokenSpec{{DepIn: []string{"mark", "advmod"}}}}
	require.Equal(t, []int{0, 1}, matchPattern(d, p3))
}

// TestLoaderRecognisesIsSpace — md/lg ship three patterns using the
// boolean IS_SPACE flag (en_core_web_md patterns 176-178). Loader maps
// {IS_SPACE: true} → IsSpace=1 and {IS_SPACE: false} → IsSpace=-1; both
// must load with Unsupported=false.
func TestLoaderRecognisesIsSpace(t *testing.T) {
	patBytes, err := msgpack.Marshal([]map[string]any{
		{
			"patterns": []any{[]any{map[string]any{"IS_SPACE": true}}},
			"attrs":    map[string]any{"TAG": "_SP", "POS": "SPACE"},
			"index":    int64(0),
		},
		{
			"patterns": []any{[]any{map[string]any{"TAG": "_SP", "IS_SPACE": false}}},
			"attrs":    map[string]any{"POS": "X"},
			"index":    int64(0),
		},
	})
	require.NoError(t, err)
	tmp := t.TempDir()
	path := filepath.Join(tmp, "patterns")
	require.NoError(t, os.WriteFile(path, patBytes, 0o644))

	pats, err := loadAttributeRulerPatterns(path)
	require.NoError(t, err)
	require.Len(t, pats, 2)
	require.False(t, pats[0].Unsupported)
	require.False(t, pats[1].Unsupported)
	require.Equal(t, int8(1), pats[0].TokenSpecs[0].IsSpace)
	require.Equal(t, int8(-1), pats[1].TokenSpecs[0].IsSpace)
	require.Equal(t, "_SP", pats[1].TokenSpecs[0].Tag)
}

// TestLoaderRecognisesDepNotIn — md pattern 177 carries
// DEP: {NOT_IN: [”]} ("Dep is non-empty"). Loader maps that to a
// DepNotIn slice and Unsupported=false.
func TestLoaderRecognisesDepNotIn(t *testing.T) {
	patBytes, err := msgpack.Marshal([]map[string]any{{
		"patterns": []any{[]any{map[string]any{
			"IS_SPACE": true,
			"DEP":      map[string]any{"NOT_IN": []any{""}},
		}}},
		"attrs": map[string]any{"DEP": "dep"},
		"index": int64(0),
	}})
	require.NoError(t, err)
	tmp := t.TempDir()
	path := filepath.Join(tmp, "patterns")
	require.NoError(t, os.WriteFile(path, patBytes, 0o644))

	pats, err := loadAttributeRulerPatterns(path)
	require.NoError(t, err)
	require.Len(t, pats, 1)
	require.False(t, pats[0].Unsupported)
	require.Equal(t, int8(1), pats[0].TokenSpecs[0].IsSpace)
	require.Equal(t, []string{""}, pats[0].TokenSpecs[0].DepNotIn)
}

// TestMatcher_IsSpaceAndDepNotIn_Match — direct matchPattern exercise of
// the IS_SPACE tri-state and DepNotIn sentinel. Reproduces the three
// md patterns by hand and asserts each fires on the right token only.
func TestMatcher_IsSpaceAndDepNotIn_Match(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	tagSP := ss.Add("_SP")
	tagNN := ss.Add("NN")
	depPunct := ss.Add("punct")

	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "\n  ", Tag: tagSP},             // tok 0: whitespace, no Dep
		{Text: "hello", Tag: tagNN},            // tok 1: not whitespace
		{Text: " ", Tag: tagSP, Dep: depPunct}, // tok 2: whitespace + Dep set
	}}

	// Pattern A (md #178): {IS_SPACE: True} → matches tok 0 and tok 2.
	pA := arPattern{TokenSpecs: []arTokenSpec{{IsSpace: 1}}}
	require.Equal(t, []int{0, 2}, matchPattern(d, pA))

	// Pattern B (md #176): {TAG: _SP, IS_SPACE: False} → matches no token
	// (every _SP token in this fixture IS whitespace).
	pB := arPattern{TokenSpecs: []arTokenSpec{{Tag: "_SP", IsSpace: -1}}}
	require.Empty(t, matchPattern(d, pB))

	// Pattern C (md #177): {IS_SPACE: True, DEP: {NOT_IN: ['']}} →
	// tok 0 fails (Dep == 0, sentinel blocks); tok 2 passes (Dep set).
	pC := arPattern{TokenSpecs: []arTokenSpec{{IsSpace: 1, DepNotIn: []string{""}}}}
	require.Equal(t, []int{2}, matchPattern(d, pC))
}

// TestLoaderRecognisesLowerRegex — md/lg patterns 147/153/163 use
// LOWER: {REGEX: "..."} for English contractions. Loader compiles the
// regex into LowerRegex with Unsupported=false; matchPattern matches
// against strings.ToLower(tok.Text).
func TestLoaderRecognisesLowerRegex(t *testing.T) {
	patBytes, err := msgpack.Marshal([]map[string]any{{
		"patterns": []any{[]any{map[string]any{
			"LOWER": map[string]any{"REGEX": "^n'?t$"},
		}}},
		"attrs": map[string]any{"POS": "PART"},
		"index": int64(0),
	}})
	require.NoError(t, err)
	tmp := t.TempDir()
	path := filepath.Join(tmp, "patterns")
	require.NoError(t, os.WriteFile(path, patBytes, 0o644))

	pats, err := loadAttributeRulerPatterns(path)
	require.NoError(t, err)
	require.Len(t, pats, 1)
	require.False(t, pats[0].Unsupported)
	require.NotNil(t, pats[0].TokenSpecs[0].LowerRegex)
	// Sanity-check the compiled regex matches the contraction form.
	require.True(t, pats[0].TokenSpecs[0].LowerRegex.MatchString("n't"))
	require.True(t, pats[0].TokenSpecs[0].LowerRegex.MatchString("nt"))
	require.False(t, pats[0].TokenSpecs[0].LowerRegex.MatchString("can"))
}

// TestMatcher_LowerRegex_Match — exercise the regex arm of matchPattern.
// "ain't" tokenises to ["ai", "n't"]; pattern matches the second token only.
func TestMatcher_LowerRegex_Match(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "ai", Lower: ss.Add("ai")},
		{Text: "n't", Lower: ss.Add("n't")},
		{Text: "Can", Lower: ss.Add("can")}, // uppercase Text → tests strings.ToLower path
	}}

	patBytes, err := msgpack.Marshal([]map[string]any{{
		"patterns": []any{[]any{map[string]any{
			"LOWER": map[string]any{"REGEX": "^n'?t$"},
		}}},
		"attrs": map[string]any{"POS": "PART"},
		"index": int64(0),
	}})
	require.NoError(t, err)
	tmp := t.TempDir()
	path := filepath.Join(tmp, "patterns")
	require.NoError(t, os.WriteFile(path, patBytes, 0o644))
	pats, err := loadAttributeRulerPatterns(path)
	require.NoError(t, err)

	require.Equal(t, []int{1}, matchPattern(d, pats[0]))
}

// TestLoaderRecognisesDepInAndTagNotIn handles the {NOT_IN} negation form
// and the {IN: ...} dep list.
func TestLoaderRecognisesDepInAndTagNotIn(t *testing.T) {
	patBytes, err := msgpack.Marshal([]map[string]any{{
		"patterns": []any{[]any{map[string]any{
			"TAG": map[string]any{"NOT_IN": []any{"TO", ""}},
			"DEP": map[string]any{"IN": []any{"aux", "auxpass"}},
		}}},
		"attrs": map[string]any{"POS": "AUX"},
		"index": int64(0),
	}})
	require.NoError(t, err)
	tmp := t.TempDir()
	path := filepath.Join(tmp, "patterns")
	require.NoError(t, os.WriteFile(path, patBytes, 0o644))

	pats, err := loadAttributeRulerPatterns(path)
	require.NoError(t, err)
	require.Len(t, pats, 1)
	require.False(t, pats[0].Unsupported)
	require.Equal(t, []string{"TO", ""}, pats[0].TokenSpecs[0].TagNotIn)
	require.Equal(t, []string{"aux", "auxpass"}, pats[0].TokenSpecs[0].DepIn)
}
