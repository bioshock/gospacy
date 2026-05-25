package en

import (
	"testing"

	"github.com/dlclark/regexp2"
	"github.com/stretchr/testify/require"
)

func TestPunctuation_PatternsCompile(t *testing.T) {
	for i, p := range Prefixes {
		_, err := regexp2.Compile(p, 0)
		require.NoErrorf(t, err, "Prefixes[%d] = %q", i, p)
	}
	for i, p := range Suffixes {
		_, err := regexp2.Compile(p, 0)
		require.NoErrorf(t, err, "Suffixes[%d] = %q", i, p)
	}
	for i, p := range Infixes {
		_, err := regexp2.Compile(p, 0)
		require.NoErrorf(t, err, "Infixes[%d] = %q", i, p)
	}
}

func TestPunctuation_Counts(t *testing.T) {
	require.GreaterOrEqual(t, len(Prefixes), 5, "expected at least 5 prefix patterns")
	require.GreaterOrEqual(t, len(Suffixes), 5, "expected at least 5 suffix patterns")
	require.GreaterOrEqual(t, len(Infixes), 3, "expected at least 3 infix patterns")
}
