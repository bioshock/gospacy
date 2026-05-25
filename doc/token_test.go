package doc

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToken_ZeroValueIsSafe(t *testing.T) {
	var tok Token
	require.Equal(t, "", tok.Text)
	require.Equal(t, "", tok.Whitespace)
	require.Equal(t, "", tok.Shape)
	require.Equal(t, "", tok.Morph)
	require.Equal(t, uint64(0), tok.Orth)
	require.Equal(t, int8(0), tok.SentStart)
	require.Equal(t, uint8(0), tok.EntIOB)
}

func TestToken_TextWithWhitespace(t *testing.T) {
	tok := Token{Text: "hello", Whitespace: " "}
	require.Equal(t, "hello ", tok.TextWithWS())
}
