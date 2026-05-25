package en

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMakeRules_Builds(t *testing.T) {
	r, err := MakeRules()
	require.NoError(t, err)
	require.NotNil(t, r)
}

func TestMakeRules_KnownContraction(t *testing.T) {
	r, err := MakeRules()
	require.NoError(t, err)
	pieces, ok := r.Special("don't")
	require.True(t, ok, `expected "don't" in English specials`)
	require.GreaterOrEqual(t, len(pieces), 2)
}
