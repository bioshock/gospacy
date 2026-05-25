package doc

import (
	"testing"

	"github.com/bioshock/gospacy/v3/vocab"
	"github.com/stretchr/testify/require"
)

func TestDoc_NewEmpty(t *testing.T) {
	v := vocab.NewVocab()
	d := NewDoc(v, "")
	require.Equal(t, 0, d.NumTokens())
	require.Equal(t, "", d.Text())
	require.Same(t, v, d.Vocab)
}

func TestDoc_TextRoundTrip(t *testing.T) {
	v := vocab.NewVocab()
	d := NewDoc(v, "Hello world.")
	d.Tokens = []Token{
		{Text: "Hello", Whitespace: " ", Idx: 0},
		{Text: "world", Whitespace: "", Idx: 6},
		{Text: ".", Whitespace: "", Idx: 11},
	}
	require.Equal(t, 3, d.NumTokens())
	require.Equal(t, "Hello world.", d.Text())
}

func TestDoc_TextRoundTripPreservesAllWhitespace(t *testing.T) {
	v := vocab.NewVocab()
	src := "a\tb\n c"
	d := NewDoc(v, src)
	d.Tokens = []Token{
		{Text: "a", Whitespace: "\t", Idx: 0},
		{Text: "b", Whitespace: "\n ", Idx: 2},
		{Text: "c", Whitespace: "", Idx: 5},
	}
	require.Equal(t, src, d.Text())
}
