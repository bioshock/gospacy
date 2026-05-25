package tokenizer_test

import (
	"testing"

	"github.com/bioshock/gospacy/v3/lang/en"
	"github.com/bioshock/gospacy/v3/tokenizer"
	"github.com/bioshock/gospacy/v3/vocab"
	"github.com/stretchr/testify/require"
)

func TestTokenizer_ToDoc(t *testing.T) {
	rules, err := en.MakeRules()
	require.NoError(t, err)
	tk := tokenizer.New(rules)
	v := vocab.NewVocab()
	d := tk.ToDoc(v, "Hello world.")
	require.Equal(t, 3, d.NumTokens())
	require.Equal(t, "Hello", d.Tokens[0].Text)
	require.Equal(t, " ", d.Tokens[0].Whitespace)
	require.Equal(t, "world", d.Tokens[1].Text)
	require.Equal(t, "", d.Tokens[1].Whitespace)
	require.Equal(t, ".", d.Tokens[2].Text)
	require.Equal(t, "Hello world.", d.Text())
	// Orth must be interned in Vocab.
	require.NotEqual(t, uint64(0), d.Tokens[0].Orth)
	got, ok := v.StringStore().Lookup(d.Tokens[0].Orth)
	require.True(t, ok)
	require.Equal(t, "Hello", got)
}
