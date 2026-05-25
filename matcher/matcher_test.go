package matcher

import (
	"regexp"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// TestMatcher_Add_ValidationErrors — Add rejects empty key, empty
// pattern list, empty pattern, conflicting attribute constraints,
// and out-of-range tri-state values. Locks the public contract.
func TestMatcher_Add_ValidationErrors(t *testing.T) {
	v := vocab.NewVocab()
	m := New(v)

	require.Error(t, m.Add(""),
		"empty key must error")
	require.Error(t, m.Add("KEY"),
		"no patterns must error")
	require.Error(t, m.Add("KEY", nil),
		"empty pattern must error")
	require.Error(t, m.Add("KEY",
		[]TokenSpec{{Lower: "x", LowerIn: []string{"y"}}}),
		"Lower + LowerIn must error")
	require.Error(t, m.Add("KEY",
		[]TokenSpec{{Tag: "NN", TagIn: []string{"VB"}}}),
		"Tag + TagIn must error")
	require.Error(t, m.Add("KEY",
		[]TokenSpec{{TagIn: []string{"NN"}, TagNotIn: []string{"VB"}}}),
		"TagIn + TagNotIn must error")
	require.Error(t, m.Add("KEY",
		[]TokenSpec{{IsAlpha: 2}}),
		"tri-state out of {-1,0,1} must error")
}

// TestMatcher_Add_ValidShapes — every supported "valid" spec shape
// passes Add without error. Smoke-check on each attr family.
func TestMatcher_Add_ValidShapes(t *testing.T) {
	v := vocab.NewVocab()
	m := New(v)

	require.NoError(t, m.Add("ORTH", []TokenSpec{{Orth: "Apple"}}))
	require.NoError(t, m.Add("LOWER_SET", []TokenSpec{{LowerIn: []string{"a", "b"}}}))
	require.NoError(t, m.Add("LOWER_RX", []TokenSpec{{LowerRegex: regexp.MustCompile("^x$")}}))
	require.NoError(t, m.Add("TAG_NOT_IN", []TokenSpec{{TagNotIn: []string{""}}}))
	require.NoError(t, m.Add("POS", []TokenSpec{{Pos: "NOUN"}}))
	require.NoError(t, m.Add("LEMMA_SET", []TokenSpec{{LemmaIn: []string{"run", "walk"}}}))
	require.NoError(t, m.Add("ENT", []TokenSpec{{EntType: "PERSON"}}))
	require.NoError(t, m.Add("FLAGS", []TokenSpec{{IsAlpha: 1, IsStop: -1}}))
}

// TestMatcher_Remove — Remove drops the entire key; subsequent Matches
// doesn't see it. No-op on absent keys.
func TestMatcher_Remove(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "Apple", Lower: ss.Add("apple")},
	}}

	m := New(v)
	require.NoError(t, m.Add("FRUIT", []TokenSpec{{Lower: "apple"}}))
	require.Len(t, m.Matches(d), 1)

	m.Remove("FRUIT")
	require.Empty(t, m.Matches(d))

	// No-op on absent key.
	m.Remove("NEVER_ADDED")
}

// TestMatcher_OrthEquality — Orth scalar must match exact orth hash.
func TestMatcher_OrthEquality(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "Apple", Orth: ss.Add("Apple")},
		{Text: "buys", Orth: ss.Add("buys")},
	}}

	m := New(v)
	require.NoError(t, m.Add("APPLE", []TokenSpec{{Orth: "Apple"}}))

	hits := m.Matches(d)
	require.Len(t, hits, 1)
	require.Equal(t, Match{Key: "APPLE", Start: 0, End: 1}, hits[0])
}

// TestMatcher_LowerSetMembership — LowerIn matches multiple alternatives.
func TestMatcher_LowerSetMembership(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "Artificial", Lower: ss.Add("artificial")},
		{Text: "Intelligence", Lower: ss.Add("intelligence")},
		{Text: "AI", Lower: ss.Add("ai")},
	}}

	m := New(v)
	require.NoError(t, m.Add("AI_ANY",
		[]TokenSpec{{LowerIn: []string{"artificial", "ai"}}}))

	hits := m.Matches(d)
	require.Len(t, hits, 2)
	require.Equal(t, 0, hits[0].Start)
	require.Equal(t, 2, hits[1].Start)
}

// TestMatcher_MultiTokenSequence — two-spec pattern matches a
// contiguous run; partial matches don't fire.
func TestMatcher_MultiTokenSequence(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "Artificial", Lower: ss.Add("artificial")},
		{Text: "Intelligence", Lower: ss.Add("intelligence")},
		{Text: "is", Lower: ss.Add("is")},
		{Text: "neat", Lower: ss.Add("neat")},
	}}

	m := New(v)
	require.NoError(t, m.Add("AI_PHRASE",
		[]TokenSpec{
			{Lower: "artificial"},
			{Lower: "intelligence"},
		}))

	hits := m.Matches(d)
	require.Len(t, hits, 1)
	require.Equal(t, Match{Key: "AI_PHRASE", Start: 0, End: 2}, hits[0])
}

// TestMatcher_TagNotIn_WithEmptySentinel — TagNotIn:[""] blocks
// tokens whose Tag is unset (sentinel semantics).
func TestMatcher_TagNotIn_WithEmptySentinel(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "untagged"}, // Tag=0
		{Text: "tagged", Tag: ss.Add("NN")},
	}}

	m := New(v)
	require.NoError(t, m.Add("HAS_TAG",
		[]TokenSpec{{TagNotIn: []string{""}}}))

	hits := m.Matches(d)
	require.Len(t, hits, 1)
	require.Equal(t, 1, hits[0].Start)
}

// TestMatcher_LowerRegex — REGEX matches strings.ToLower(Text).
func TestMatcher_LowerRegex(t *testing.T) {
	v := vocab.NewVocab()
	rx := regexp.MustCompile("^n'?t$")
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "I"},
		{Text: "N'T"}, // mixed-case to test lowercasing
		{Text: "not"},
	}}

	m := New(v)
	require.NoError(t, m.Add("NEG", []TokenSpec{{LowerRegex: rx}}))

	hits := m.Matches(d)
	require.Len(t, hits, 1)
	require.Equal(t, 1, hits[0].Start)
}

// TestMatcher_PosAndLemma — POS + LEMMA constraints conjunction.
func TestMatcher_PosAndLemma(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "runs", POS: ss.Add("VERB"), Lemma: ss.Add("run")},
		{Text: "fast", POS: ss.Add("ADV"), Lemma: ss.Add("fast")},
		{Text: "ran", POS: ss.Add("VERB"), Lemma: ss.Add("run")},
	}}

	m := New(v)
	require.NoError(t, m.Add("RUN_VERB",
		[]TokenSpec{{Pos: "VERB", Lemma: "run"}}))

	hits := m.Matches(d)
	require.Len(t, hits, 2)
	require.Equal(t, 0, hits[0].Start)
	require.Equal(t, 2, hits[1].Start)
}

// TestMatcher_EntTypeSet — EntTypeIn fires on tokens inside a named
// entity. Useful for "match all PERSON tokens".
func TestMatcher_EntTypeSet(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "Tim", EntType: ss.Add("PERSON")},
		{Text: "Cook", EntType: ss.Add("PERSON")},
		{Text: "spoke"},
	}}

	m := New(v)
	require.NoError(t, m.Add("PEOPLE",
		[]TokenSpec{{EntTypeIn: []string{"PERSON", "ORG"}}}))

	hits := m.Matches(d)
	require.Len(t, hits, 2)
}

// TestMatcher_IsAlpha_TriState — IS_ALPHA: true and false polarities.
func TestMatcher_IsAlpha_TriState(t *testing.T) {
	v := vocab.NewVocab()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "hello"},
		{Text: "42"},
		{Text: "Apple"},
	}}

	m := New(v)
	require.NoError(t, m.Add("LETTERS", []TokenSpec{{IsAlpha: 1}}))
	require.NoError(t, m.Add("NOT_LETTERS", []TokenSpec{{IsAlpha: -1}}))

	hits := m.Matches(d)
	// LETTERS fires on tok 0 and 2; NOT_LETTERS fires on tok 1.
	// Output is sorted by (Start, End, Key).
	require.Len(t, hits, 3)
	require.Equal(t, Match{Key: "LETTERS", Start: 0, End: 1}, hits[0])
	require.Equal(t, Match{Key: "NOT_LETTERS", Start: 1, End: 2}, hits[1])
	require.Equal(t, Match{Key: "LETTERS", Start: 2, End: 3}, hits[2])
}

// TestMatcher_LikeNum — LIKE_NUM covers digits, signed, words, ordinal,
// suffix-form. Exhaustive coverage lives in internal/lexflags;
// here we just verify the matcher hooks into it.
func TestMatcher_LikeNum(t *testing.T) {
	v := vocab.NewVocab()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "42"},
		{Text: "twenty"},
		{Text: "21st"},
		{Text: "hello"},
	}}

	m := New(v)
	require.NoError(t, m.Add("NUM", []TokenSpec{{LikeNum: 1}}))

	hits := m.Matches(d)
	require.Len(t, hits, 3)
	require.Equal(t, 0, hits[0].Start)
	require.Equal(t, 1, hits[1].Start)
	require.Equal(t, 2, hits[2].Start)
}

// TestMatcher_IsSpace — IS_SPACE tri-state.
func TestMatcher_IsSpace(t *testing.T) {
	v := vocab.NewVocab()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "hello"},
		{Text: "\n  "},
		{Text: "world"},
	}}

	m := New(v)
	require.NoError(t, m.Add("WS", []TokenSpec{{IsSpace: 1}}))

	hits := m.Matches(d)
	require.Len(t, hits, 1)
	require.Equal(t, 1, hits[0].Start)
}

// TestMatcher_NoMatch — returns empty (not nil-after-non-nil
// allocation) when nothing matches.
func TestMatcher_NoMatch(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "Apple", Orth: ss.Add("Apple")},
	}}

	m := New(v)
	require.NoError(t, m.Add("ORANGE", []TokenSpec{{Orth: "Orange"}}))

	hits := m.Matches(d)
	require.Empty(t, hits)
}

// TestMatcher_OverlapDedup_SameKeyLongestWins — within a single key,
// when two alternatives overlap, the longer span wins. Same-position
// overlaps across DIFFERENT keys are preserved (callers may want
// both labels).
func TestMatcher_OverlapDedup_SameKeyLongestWins(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "Artificial", Lower: ss.Add("artificial")},
		{Text: "Intelligence", Lower: ss.Add("intelligence")},
	}}

	m := New(v)
	// Same key, two alternatives — the 2-token span subsumes the
	// 1-token span; only the longer should survive.
	require.NoError(t, m.Add("AI",
		[]TokenSpec{{Lower: "artificial"}, {Lower: "intelligence"}},
		[]TokenSpec{{Lower: "artificial"}},
	))
	hits := m.Matches(d)
	require.Len(t, hits, 1)
	require.Equal(t, Match{Key: "AI", Start: 0, End: 2}, hits[0])
}

// TestMatcher_OverlapDedup_CrossKeyKept — overlap dedup is per-key.
// Two different keys covering the same span both fire.
func TestMatcher_OverlapDedup_CrossKeyKept(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "Apple", Lower: ss.Add("apple")},
	}}

	m := New(v)
	require.NoError(t, m.Add("FRUIT", []TokenSpec{{Lower: "apple"}}))
	require.NoError(t, m.Add("COMPANY", []TokenSpec{{Lower: "apple"}}))

	hits := m.Matches(d)
	require.Len(t, hits, 2)
	require.Equal(t, []string{"COMPANY", "FRUIT"}, []string{hits[0].Key, hits[1].Key})
}

// TestMatcher_AlternativesUnion — same key with two alternatives;
// both fire and end up in the result set.
func TestMatcher_AlternativesUnion(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{Text: "AI", Lower: ss.Add("ai")},
		{Text: "ML", Lower: ss.Add("ml")},
	}}

	m := New(v)
	require.NoError(t, m.Add("ABBREV",
		[]TokenSpec{{Lower: "ai"}},
		[]TokenSpec{{Lower: "ml"}},
	))

	hits := m.Matches(d)
	require.Len(t, hits, 2)
	// Both under same key.
	require.Equal(t, "ABBREV", hits[0].Key)
	require.Equal(t, "ABBREV", hits[1].Key)
}
