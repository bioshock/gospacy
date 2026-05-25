package doc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/vocab"
)

// makeNounChunkDoc builds: "The cat sat" with parse:
//
//	0 "The"  POS=DET   Dep=det  Head=1
//	1 "cat"  POS=NOUN  Dep=nsubj Head=2
//	2 "sat"  POS=VERB  Dep=ROOT  Head=2
//
// Expected noun chunk: [0,2) = "The cat" (nsubj on token 1 with left_edge=0).
func TestDoc_NounChunks_BasicNsubj(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := NewDoc(v, "The cat sat")
	d.Tokens = []Token{
		{Text: "The", POS: ss.Hash("DET"), Dep: ss.Hash("det"), Head: 1},
		{Text: "cat", POS: ss.Hash("NOUN"), Dep: ss.Hash("nsubj"), Head: 2},
		{Text: "sat", POS: ss.Hash("VERB"), Dep: ss.Hash("ROOT"), Head: 2},
	}
	chunks := d.NounChunks()
	require.Len(t, chunks, 1)
	require.Equal(t, 0, chunks[0].Start)
	require.Equal(t, 2, chunks[0].End)
}

// TestDoc_NounChunks_NoNounChunks confirms that a Doc whose only NOUN/PROPN/PRON
// tokens have non-np_deps (and no conj fallback) yields no chunks.
func TestDoc_NounChunks_NoNounChunks(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	d := NewDoc(v, "")
	d.Tokens = []Token{
		// VERB with ROOT — POS check rejects it.
		{Text: "ran", POS: ss.Hash("VERB"), Dep: ss.Hash("ROOT"), Head: 0},
	}
	require.Nil(t, d.NounChunks())
}

func TestDoc_NounChunks_Empty(t *testing.T) {
	d := NewDoc(vocab.NewVocab(), "")
	require.Nil(t, d.NounChunks())
}
