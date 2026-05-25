package vocab

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestVocab_GetCaches(t *testing.T) {
	v := NewVocab()
	lex1 := v.Get("hello")
	lex2 := v.Get("hello")
	require.Same(t, lex1, lex2, "Vocab.Get must return the same pointer for repeated calls")
}

func TestVocab_GetByHash(t *testing.T) {
	v := NewVocab()
	lex := v.Get("hello")
	got := v.GetByHash(lex.Orth)
	require.Same(t, lex, got)
}

func TestVocab_GetByHash_NotInterned(t *testing.T) {
	v := NewVocab()
	got := v.GetByHash(uint64(0xdeadbeefcafef00d))
	require.Nil(t, got)
}

func TestVocab_StringStoreAccessor(t *testing.T) {
	v := NewVocab()
	v.Get("x")
	require.NotNil(t, v.StringStore())
	_, ok := v.StringStore().Get("x")
	require.True(t, ok)
}
