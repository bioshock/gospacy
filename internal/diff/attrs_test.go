package diff

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompareAttrs_AllEqual(t *testing.T) {
	want := []TokenAttr{{Orth: "Apple", Tag: "NNP", POS: "PROPN", Morph: "Number=Sing", Lemma: "Apple"}}
	got := []TokenAttr{{Orth: "Apple", Tag: "NNP", POS: "PROPN", Morph: "Number=Sing", Lemma: "Apple"}}
	rep := CompareAttrs(want, got)
	require.True(t, rep.Equal())
}

func TestCompareAttrs_TagMismatch(t *testing.T) {
	want := []TokenAttr{{Orth: "ran", Tag: "VBD", POS: "VERB", Lemma: "run"}}
	got := []TokenAttr{{Orth: "ran", Tag: "VBN", POS: "VERB", Lemma: "run"}}
	rep := CompareAttrs(want, got)
	require.False(t, rep.Equal())
	require.Equal(t, AttrTag, rep.AttrDisagreements[0].Attr)
	require.Equal(t, 0, rep.AttrDisagreements[0].Index)
}

func TestCompareAttrs_MultipleDisagreements(t *testing.T) {
	want := []TokenAttr{{Orth: "x", Tag: "A", POS: "X", Morph: "M=1", Lemma: "x"}}
	got := []TokenAttr{{Orth: "x", Tag: "B", POS: "Y", Morph: "M=2", Lemma: "y"}}
	rep := CompareAttrs(want, got)
	require.Len(t, rep.AttrDisagreements, 4)
}
