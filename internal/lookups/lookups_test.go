package lookups

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLookups_LoadLemmatizerBundle(t *testing.T) {
	path := "../../testdata/models/en_core_web_sm/lemmatizer/lookups/lookups.bin"
	if _, err := os.Stat(path); err != nil {
		t.Skip("model not downloaded")
	}
	l, err := Load(path)
	require.NoError(t, err)
	// en_core_web_sm ships these three tables.
	require.True(t, l.Has("lemma_rules"))
	require.True(t, l.Has("lemma_exc"))
	require.True(t, l.Has("lemma_index"))
	require.False(t, l.Has("does_not_exist"))
}

func TestLookups_TableGetByHash(t *testing.T) {
	path := "../../testdata/models/en_core_web_sm/lemmatizer/lookups/lookups.bin"
	if _, err := os.Stat(path); err != nil {
		t.Skip("model not downloaded")
	}
	l, err := Load(path)
	require.NoError(t, err)
	rules := l.Get("lemma_rules")
	// Hash for "verb" is 12401032943472870168 per the python probe.
	v, ok := rules.GetByHash(12401032943472870168)
	require.True(t, ok)
	require.NotNil(t, v)
}

func TestLookups_LoadVocabBundle(t *testing.T) {
	path := "../../testdata/models/en_core_web_sm/vocab/lookups.bin"
	if _, err := os.Stat(path); err != nil {
		t.Skip("model not downloaded")
	}
	l, err := Load(path)
	require.NoError(t, err)
	require.True(t, l.Has("lexeme_norm"))
}
