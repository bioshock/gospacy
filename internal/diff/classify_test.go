package diff

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestClassify_LengthOnly(t *testing.T) {
	rep := Report{Disagreements: []Disagreement{{Kind: DisagreeLength, Index: -1}}}
	c := Classify(rep)
	require.Equal(t, ClassLengthMismatch, c.Primary)
}

func TestClassify_WhitespaceOnly(t *testing.T) {
	rep := Report{Disagreements: []Disagreement{
		{Kind: DisagreeWS, Index: 1, Want: "true", Got: "false"},
		{Kind: DisagreeWS, Index: 2, Want: "false", Got: "true"},
	}}
	c := Classify(rep)
	require.Equal(t, ClassWhitespaceOnly, c.Primary)
}

func TestClassify_OrthDifference(t *testing.T) {
	rep := Report{Disagreements: []Disagreement{
		{Kind: DisagreeOrth, Index: 3, Want: "don't", Got: "dont"},
	}}
	c := Classify(rep)
	require.Equal(t, ClassRealDivergence, c.Primary)
}

func TestClassify_Mixed(t *testing.T) {
	rep := Report{Disagreements: []Disagreement{
		{Kind: DisagreeOrth, Index: 3, Want: "x", Got: "y"},
		{Kind: DisagreeIdx, Index: 4, Want: "10", Got: "11"},
	}}
	c := Classify(rep)
	require.Equal(t, ClassRealDivergence, c.Primary)
	require.Equal(t, 1, c.Counts[DisagreeOrth])
	require.Equal(t, 1, c.Counts[DisagreeIdx])
}

func TestClassify_Empty(t *testing.T) {
	c := Classify(Report{})
	require.Equal(t, ClassEqual, c.Primary)
}
