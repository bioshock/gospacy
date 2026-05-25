package doc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/vocab"
)

func TestToken_IsStop_SpotCheckCommonStopwords(t *testing.T) {
	v := vocab.NewVocab()
	for _, w := range []string{"the", "of", "and", "is", "are", "was", "be", "to", "a", "in"} {
		lex := v.Get(w)
		tok := Token{Lower: lex.Lower}
		require.Truef(t, tok.IsStop(v), "expected %q to be a stop word", w)
	}
}

func TestToken_IsStop_FalseForContentWord(t *testing.T) {
	v := vocab.NewVocab()
	lex := v.Get("apple")
	tok := Token{Lower: lex.Lower}
	require.False(t, tok.IsStop(v))
}

func TestToken_IsStop_NilVocabSafe(t *testing.T) {
	require.False(t, Token{Lower: 42}.IsStop(nil))
}

func TestToken_IsStop_UnresolvableHashFalse(t *testing.T) {
	v := vocab.NewVocab()
	tok := Token{Lower: 0xdeadbeef}
	require.False(t, tok.IsStop(v))
}
