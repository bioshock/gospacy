package en

import (
	"testing"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/stretchr/testify/require"
)

func TestIsBaseForm_NounSingular(t *testing.T) {
	tok := &doc.Token{Morph: "Number=Sing"}
	require.True(t, IsBaseForm(tok, "noun"))
}

func TestIsBaseForm_VerbInf(t *testing.T) {
	tok := &doc.Token{Morph: "VerbForm=Inf"}
	require.True(t, IsBaseForm(tok, "verb"))
}

func TestIsBaseForm_VerbFinPres_NoNumber(t *testing.T) {
	tok := &doc.Token{Morph: "VerbForm=Fin|Tense=Pres"}
	require.True(t, IsBaseForm(tok, "verb"))
}

func TestIsBaseForm_VerbFinPresSing_False(t *testing.T) {
	tok := &doc.Token{Morph: "VerbForm=Fin|Tense=Pres|Number=Sing"}
	require.False(t, IsBaseForm(tok, "verb"))
}

func TestIsBaseForm_AdjPosDegree(t *testing.T) {
	tok := &doc.Token{Morph: "Degree=Pos"}
	require.True(t, IsBaseForm(tok, "adj"))
}

func TestIsBaseForm_VerbFormNoneFallback(t *testing.T) {
	tok := &doc.Token{Morph: "VerbForm=None"}
	require.True(t, IsBaseForm(tok, "x")) // any pos
}

func TestIsBaseForm_Empty(t *testing.T) {
	tok := &doc.Token{}
	require.False(t, IsBaseForm(tok, "noun"))
}
