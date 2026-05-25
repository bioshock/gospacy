package tokenizer

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func mustRules(t *testing.T) *Rules {
	t.Helper()
	r, err := NewRules(RulesInput{
		Prefixes: []string{`^\(`, `^"`, `^\$`, `^¿`},
		Suffixes: []string{`\)$`, `\.$`, `,$`, `"$`},
		Infixes:  []string{`(?<=[a-z])-(?=[a-z])`},
		Specials: map[string][]SpecialPiece{
			"don't": {{Orth: "do"}, {Orth: "n't"}},
			"U.S.":  {{Orth: "U.S."}},
		},
	})
	require.NoError(t, err)
	return r
}

func TestTokenizer_Whitespace(t *testing.T) {
	tk := New(mustRules(t))
	toks := tk.Tokenize("hello world")
	require.Len(t, toks, 2)
	require.Equal(t, "hello", toks[0].Orth)
	require.True(t, toks[0].WS)
	require.Equal(t, 0, toks[0].Idx)
	require.Equal(t, "world", toks[1].Orth)
	require.False(t, toks[1].WS)
	require.Equal(t, 6, toks[1].Idx)
}

func TestTokenizer_PrefixStrip(t *testing.T) {
	tk := New(mustRules(t))
	toks := tk.Tokenize("(hello)")
	require.Len(t, toks, 3)
	require.Equal(t, "(", toks[0].Orth)
	require.Equal(t, "hello", toks[1].Orth)
	require.Equal(t, ")", toks[2].Orth)
}

func TestTokenizer_SuffixStrip(t *testing.T) {
	tk := New(mustRules(t))
	toks := tk.Tokenize("end.")
	require.Len(t, toks, 2)
	require.Equal(t, "end", toks[0].Orth)
	require.Equal(t, ".", toks[1].Orth)
}

func TestTokenizer_Special(t *testing.T) {
	tk := New(mustRules(t))
	toks := tk.Tokenize("don't")
	require.Len(t, toks, 2)
	require.Equal(t, "do", toks[0].Orth)
	require.Equal(t, "n't", toks[1].Orth)
}

func TestTokenizer_SpecialWithSuffix(t *testing.T) {
	tk := New(mustRules(t))
	toks := tk.Tokenize("don't.")
	require.Len(t, toks, 3)
	require.Equal(t, "do", toks[0].Orth)
	require.Equal(t, "n't", toks[1].Orth)
	require.Equal(t, ".", toks[2].Orth)
}

func TestTokenizer_Infix(t *testing.T) {
	tk := New(mustRules(t))
	toks := tk.Tokenize("hello-world")
	require.Len(t, toks, 3)
	require.Equal(t, "hello", toks[0].Orth)
	require.Equal(t, "-", toks[1].Orth)
	require.Equal(t, "world", toks[2].Orth)
}

func TestTokenizer_EmptyAndWhitespaceOnly(t *testing.T) {
	tk := New(mustRules(t))
	// Empty string → no tokens.
	require.Empty(t, tk.Tokenize(""))
	// Whitespace-only → spaCy emits the whitespace run as a single token.
	toks := tk.Tokenize("   ")
	require.Len(t, toks, 1)
	require.Equal(t, "   ", toks[0].Orth)
}
