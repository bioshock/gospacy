package diff

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompareEntities_AllEqual(t *testing.T) {
	want := []Entity{{Start: 0, End: 1, Label: "ORG", Text: "Apple"}}
	got := []Entity{{Start: 0, End: 1, Label: "ORG", Text: "Apple"}}
	r := CompareEntities(want, got)
	require.True(t, r.Equal())
	require.Equal(t, 1.0, r.F1)
}

func TestCompareEntities_MissedEntity(t *testing.T) {
	// Want one entity; got none. Precision = 0/0 -> defined as 1.0, Recall = 0/1, F1 = 0.
	want := []Entity{{Start: 0, End: 1, Label: "ORG", Text: "Apple"}}
	got := []Entity{}
	r := CompareEntities(want, got)
	require.False(t, r.Equal())
	require.Equal(t, 0.0, r.Recall)
	require.Equal(t, 0.0, r.F1)
}

func TestCompareEntities_LabelMismatch(t *testing.T) {
	// Same span, wrong label -> 1 false positive + 1 false negative.
	want := []Entity{{Start: 0, End: 1, Label: "ORG"}}
	got := []Entity{{Start: 0, End: 1, Label: "PERSON"}}
	r := CompareEntities(want, got)
	require.Equal(t, 0.0, r.Precision)
	require.Equal(t, 0.0, r.Recall)
	require.Equal(t, 0.0, r.F1)
}

func TestCompareEntities_PartialOverlap(t *testing.T) {
	// 2 expected, 2 predicted, 1 exact match -> P=0.5, R=0.5, F1=0.5
	want := []Entity{{Start: 0, End: 1, Label: "ORG"}, {Start: 3, End: 5, Label: "GPE"}}
	got := []Entity{{Start: 0, End: 1, Label: "ORG"}, {Start: 3, End: 4, Label: "GPE"}}
	r := CompareEntities(want, got)
	require.Equal(t, 0.5, r.Precision)
	require.Equal(t, 0.5, r.Recall)
	require.InDelta(t, 0.5, r.F1, 1e-9)
}
