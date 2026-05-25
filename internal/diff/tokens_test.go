package diff

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompareTokens_AllEqual(t *testing.T) {
	want := []Token{{Orth: "Hello", Idx: 0, WS: true}, {Orth: "world", Idx: 6, WS: false}}
	got := []Token{{Orth: "Hello", Idx: 0, WS: true}, {Orth: "world", Idx: 6, WS: false}}
	rep := CompareTokens(want, got)
	require.True(t, rep.Equal())
	require.Empty(t, rep.Disagreements)
}

func TestCompareTokens_LengthMismatch(t *testing.T) {
	want := []Token{{Orth: "Hi", Idx: 0}, {Orth: "there", Idx: 3}}
	got := []Token{{Orth: "Hi", Idx: 0}}
	rep := CompareTokens(want, got)
	require.False(t, rep.Equal())
	require.Equal(t, 1, len(rep.Disagreements))
	require.Equal(t, DisagreeLength, rep.Disagreements[0].Kind)
}

func TestCompareTokens_OrthMismatch(t *testing.T) {
	want := []Token{{Orth: "don't", Idx: 0}}
	got := []Token{{Orth: "dont", Idx: 0}}
	rep := CompareTokens(want, got)
	require.False(t, rep.Equal())
	require.Equal(t, DisagreeOrth, rep.Disagreements[0].Kind)
	require.Equal(t, 0, rep.Disagreements[0].Index)
}

func TestCompareTokens_OffsetMismatch(t *testing.T) {
	want := []Token{{Orth: "hi", Idx: 0}, {Orth: "world", Idx: 3}}
	got := []Token{{Orth: "hi", Idx: 0}, {Orth: "world", Idx: 4}}
	rep := CompareTokens(want, got)
	require.False(t, rep.Equal())
	require.Equal(t, DisagreeIdx, rep.Disagreements[0].Kind)
}
