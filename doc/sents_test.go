package doc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/vocab"
)

func TestDoc_Sents_Empty(t *testing.T) {
	d := NewDoc(vocab.NewVocab(), "")
	require.Nil(t, d.Sents())
}

func TestDoc_Sents_SingleSentence(t *testing.T) {
	d := NewDoc(vocab.NewVocab(), "")
	d.Tokens = []Token{
		{Text: "Hi", SentStart: 1},
		{Text: "there", SentStart: 0},
		{Text: ".", SentStart: 0},
	}
	got := d.Sents()
	require.Len(t, got, 1)
	require.Equal(t, 0, got[0].Start)
	require.Equal(t, 3, got[0].End)
}

func TestDoc_Sents_TwoSentences(t *testing.T) {
	d := NewDoc(vocab.NewVocab(), "")
	d.Tokens = []Token{
		{Text: "Hi", SentStart: 1},
		{Text: ".", SentStart: 0},
		{Text: "Bye", SentStart: 1},
		{Text: ".", SentStart: 0},
	}
	got := d.Sents()
	require.Len(t, got, 2)
	require.Equal(t, 0, got[0].Start)
	require.Equal(t, 2, got[0].End)
	require.Equal(t, 2, got[1].Start)
	require.Equal(t, 4, got[1].End)
}

func TestDoc_Sents_PanicsOnUnknown(t *testing.T) {
	d := NewDoc(vocab.NewVocab(), "")
	d.Tokens = []Token{
		{Text: "Hi", SentStart: 1},
		{Text: "there", SentStart: -1},
	}
	require.Panics(t, func() { _ = d.Sents() })
}

// TestDoc_Sents_FirstTokenSentStartZero: spaCy always sets the first token's
// SentStart to 1 in a well-formed Doc. Our partition still treats index 0 as
// a sentence start even when SentStart != 1 there, because the loop only
// breaks on SentStart == 1 at i >= 1.
func TestDoc_Sents_FirstTokenSentStartZero(t *testing.T) {
	d := NewDoc(vocab.NewVocab(), "")
	d.Tokens = []Token{
		{Text: "a", SentStart: 0},
		{Text: "b", SentStart: 0},
	}
	got := d.Sents()
	require.Len(t, got, 1)
	require.Equal(t, 0, got[0].Start)
	require.Equal(t, 2, got[0].End)
}
