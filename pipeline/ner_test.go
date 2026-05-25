package pipeline

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

func TestNER_ApplyStub_TagsEveryTokenAsO(t *testing.T) {
	v := vocab.NewVocab()
	d := doc.NewDoc(v, "Apple is.")
	d.Tokens = []doc.Token{
		{Text: "Apple", Orth: v.StringStore().Add("Apple")},
		{Text: "is", Orth: v.StringStore().Add("is")},
		{Text: ".", Orth: v.StringStore().Add(".")},
	}
	n := &NER{} // zero-value — ApplyStub doesn't need a real model.
	require.NoError(t, n.ApplyStub(d))
	for _, tok := range d.Tokens {
		require.Equal(t, uint8(2), tok.EntIOB, "ApplyStub should set every token to O")
		require.Equal(t, uint64(0), tok.EntType)
	}
}
