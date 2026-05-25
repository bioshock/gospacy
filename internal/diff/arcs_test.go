package diff

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompareArcs_AllEqual(t *testing.T) {
	want := []Arc{{I: 0, Head: 1, Dep: "nsubj"}, {I: 1, Head: 1, Dep: "ROOT"}}
	got := []Arc{{I: 0, Head: 1, Dep: "nsubj"}, {I: 1, Head: 1, Dep: "ROOT"}}
	r := CompareArcs(want, got)
	require.True(t, r.Equal())
	require.Equal(t, 1.0, r.LAS) // labeled attachment score: 2/2
	require.Equal(t, 1.0, r.UAS) // unlabeled: 2/2
}

func TestCompareArcs_HeadMismatch(t *testing.T) {
	// Two tokens; token 0's head changes from 1 to 0 (LAS and UAS both 1/2)
	want := []Arc{{I: 0, Head: 1, Dep: "nsubj"}, {I: 1, Head: 1, Dep: "ROOT"}}
	got := []Arc{{I: 0, Head: 0, Dep: "nsubj"}, {I: 1, Head: 1, Dep: "ROOT"}}
	r := CompareArcs(want, got)
	require.False(t, r.Equal())
	require.Equal(t, 0.5, r.UAS)
	require.Equal(t, 0.5, r.LAS)
}

func TestCompareArcs_DepLabelMismatch(t *testing.T) {
	// Head matches but label differs: UAS=1.0, LAS=0.5
	want := []Arc{{I: 0, Head: 1, Dep: "nsubj"}, {I: 1, Head: 1, Dep: "ROOT"}}
	got := []Arc{{I: 0, Head: 1, Dep: "obj"}, {I: 1, Head: 1, Dep: "ROOT"}}
	r := CompareArcs(want, got)
	require.Equal(t, 1.0, r.UAS)
	require.Equal(t, 0.5, r.LAS)
}

func TestCompareArcs_LengthMismatch(t *testing.T) {
	want := []Arc{{I: 0, Head: 0, Dep: "ROOT"}}
	got := []Arc{}
	r := CompareArcs(want, got)
	require.False(t, r.Equal())
	require.NotNil(t, r.LengthDisagreement)
}
