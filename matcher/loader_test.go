package matcher

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// TestFromPatternDict_AllAttrs — round-trip every supported attr ×
// value form through FromPatternDict and verify the resulting
// matcher fires correctly on a hand-built doc.
func TestFromPatternDict_AllAttrs(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := &doc.Doc{Vocab: v, Tokens: []doc.Token{
		{
			Text: "Tim",
			Orth: ss.Add("Tim"), Lower: ss.Add("tim"),
			Tag: ss.Add("NNP"), POS: ss.Add("PROPN"),
			Dep: ss.Add("compound"), Lemma: ss.Add("Tim"),
			EntType: ss.Add("PERSON"),
		},
	}}

	cases := []struct {
		name string
		dict map[string]any
	}{
		{"ORTH scalar", map[string]any{"ORTH": "Tim"}},
		{"ORTH IN", map[string]any{"ORTH": map[string]any{"IN": []any{"Tim", "Cook"}}}},
		{"LOWER scalar", map[string]any{"LOWER": "tim"}},
		{"LOWER IN", map[string]any{"LOWER": map[string]any{"IN": []any{"tim"}}}},
		{"LOWER REGEX", map[string]any{"LOWER": map[string]any{"REGEX": "^t.m$"}}},
		{"TAG scalar", map[string]any{"TAG": "NNP"}},
		{"TAG IN", map[string]any{"TAG": map[string]any{"IN": []any{"NNP", "NNS"}}}},
		{"TAG NOT_IN", map[string]any{"TAG": map[string]any{"NOT_IN": []any{"VBZ"}}}},
		{"POS scalar", map[string]any{"POS": "PROPN"}},
		{"POS IN", map[string]any{"POS": map[string]any{"IN": []any{"PROPN", "NOUN"}}}},
		{"DEP scalar", map[string]any{"DEP": "compound"}},
		{"DEP IN", map[string]any{"DEP": map[string]any{"IN": []any{"compound", "nsubj"}}}},
		{"DEP NOT_IN", map[string]any{"DEP": map[string]any{"NOT_IN": []any{""}}}},
		{"LEMMA scalar", map[string]any{"LEMMA": "Tim"}},
		{"LEMMA IN", map[string]any{"LEMMA": map[string]any{"IN": []any{"Tim", "Cook"}}}},
		{"ENT_TYPE scalar", map[string]any{"ENT_TYPE": "PERSON"}},
		{"ENT_TYPE IN", map[string]any{"ENT_TYPE": map[string]any{"IN": []any{"PERSON", "ORG"}}}},
		{"IS_ALPHA true", map[string]any{"IS_ALPHA": true}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := New(v)
			require.NoError(t, m.FromPatternDict("K", [][]map[string]any{{c.dict}}))
			hits := m.Matches(d)
			require.Len(t, hits, 1, "pattern %q should match the single-Tim doc", c.name)
		})
	}
}

// TestFromPatternDict_QuantifierOpsFailLoud — OP key must error (Tier
// 2 deferred). Locks the Rule 12 "fail loud" promise.
func TestFromPatternDict_QuantifierOpsFailLoud(t *testing.T) {
	v := vocab.NewVocab()
	m := New(v)
	err := m.FromPatternDict("K", [][]map[string]any{{
		{"LOWER": "x", "OP": "?"},
	}})
	require.Error(t, err)
	require.Contains(t, err.Error(), "OP")
}

// TestFromPatternDict_UnknownKeyFailsLoud — typo'd / unsupported key
// must error rather than silently skip. Matches AR's loader-Unsupported
// philosophy but exposes it as a Go error since matcher is a public
// API.
func TestFromPatternDict_UnknownKeyFailsLoud(t *testing.T) {
	v := vocab.NewVocab()
	m := New(v)
	err := m.FromPatternDict("K", [][]map[string]any{{
		{"NORM": "x"}, // NORM not implemented
	}})
	require.Error(t, err)
	require.Contains(t, err.Error(), "unsupported pattern key")
}

// TestFromPatternDict_InvalidREGEX — malformed regex must surface
// the compile error, not silently fall through.
func TestFromPatternDict_InvalidREGEX(t *testing.T) {
	v := vocab.NewVocab()
	m := New(v)
	err := m.FromPatternDict("K", [][]map[string]any{{
		{"LOWER": map[string]any{"REGEX": "[unclosed"}},
	}})
	require.Error(t, err)
	require.Contains(t, err.Error(), "invalid REGEX")
}

// TestFromPatternDict_NonStringInIN — {IN: [1, 2]} (numbers, not
// strings) must fail loud rather than partially decode.
func TestFromPatternDict_NonStringInIN(t *testing.T) {
	v := vocab.NewVocab()
	m := New(v)
	err := m.FromPatternDict("K", [][]map[string]any{{
		{"ORTH": map[string]any{"IN": []any{1, 2}}},
	}})
	require.Error(t, err)
}

// TestFromPatternDict_BoolWrongType — IS_ALPHA: "true" (string, not
// bool) must error.
func TestFromPatternDict_BoolWrongType(t *testing.T) {
	v := vocab.NewVocab()
	m := New(v)
	err := m.FromPatternDict("K", [][]map[string]any{{
		{"IS_ALPHA": "true"}, // wrong type
	}})
	require.Error(t, err)
	require.Contains(t, err.Error(), "IS_ALPHA")
}
