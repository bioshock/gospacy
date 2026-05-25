package tokenizer

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRules_Build_Empty(t *testing.T) {
	r, err := NewRules(RulesInput{})
	require.NoError(t, err)
	require.NotNil(t, r)
	got, ok := r.FindPrefix("hello")
	require.False(t, ok)
	require.Equal(t, "", got)
}

func TestRules_FindPrefix_SimplePattern(t *testing.T) {
	r, err := NewRules(RulesInput{
		Prefixes: []string{`^\(`},
	})
	require.NoError(t, err)
	got, ok := r.FindPrefix("(hello")
	require.True(t, ok)
	require.Equal(t, "(", got)

	got, ok = r.FindPrefix("hello")
	require.False(t, ok)
	require.Equal(t, "", got)
}

func TestRules_FindSuffix(t *testing.T) {
	r, err := NewRules(RulesInput{
		Suffixes: []string{`\.$`},
	})
	require.NoError(t, err)
	got, ok := r.FindSuffix("end.")
	require.True(t, ok)
	require.Equal(t, ".", got)
}

func TestRules_FindInfixes(t *testing.T) {
	r, err := NewRules(RulesInput{
		Infixes: []string{`-`},
	})
	require.NoError(t, err)
	spans := r.FindInfixes("hello-world-x")
	require.Len(t, spans, 2)
	require.Equal(t, 5, spans[0].Start)
	require.Equal(t, 6, spans[0].End)
	require.Equal(t, 11, spans[1].Start)
	require.Equal(t, 12, spans[1].End)
}

func TestRules_SpecialCase(t *testing.T) {
	r, err := NewRules(RulesInput{
		Specials: map[string][]SpecialPiece{
			"don't": {{Orth: "do"}, {Orth: "n't"}},
		},
	})
	require.NoError(t, err)
	pieces, ok := r.Special("don't")
	require.True(t, ok)
	require.Len(t, pieces, 2)
	require.Equal(t, "do", pieces[0].Orth)
	require.Equal(t, "n't", pieces[1].Orth)

	_, ok = r.Special("nope")
	require.False(t, ok)
}
