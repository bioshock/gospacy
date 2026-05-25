package doc

import (
	"testing"

	"github.com/bioshock/gospacy/v3/vocab"
	"github.com/stretchr/testify/require"
)

func TestSpan_BasicRange(t *testing.T) {
	v := vocab.NewVocab()
	d := NewDoc(v, "Hello world.")
	d.Tokens = []Token{
		{Text: "Hello", Whitespace: " "},
		{Text: "world", Whitespace: ""},
		{Text: ".", Whitespace: ""},
	}
	sp := Span{Doc: d, Start: 0, End: 2}
	require.Equal(t, 2, sp.Len())
	require.Equal(t, "Hello world", sp.Text())
}

func TestSpan_IndexNegative(t *testing.T) {
	v := vocab.NewVocab()
	d := NewDoc(v, "a b c")
	d.Tokens = []Token{
		{Text: "a"}, {Text: "b"}, {Text: "c"},
	}
	sp := Span{Doc: d, Start: 0, End: 3}
	require.Equal(t, "a", sp.At(0).Text)
	require.Equal(t, "c", sp.At(-1).Text)
	require.Equal(t, "b", sp.At(-2).Text)
}
