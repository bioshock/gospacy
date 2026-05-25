package diff

import (
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// repoRoot returns the absolute path to the repo root so tests can locate testdata/.
func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	require.True(t, ok)
	// .../internal/diff/fixture_test.go → repo root is two dirs up from internal/diff
	return filepath.Join(filepath.Dir(file), "..", "..")
}

func TestLoadSampleTokens(t *testing.T) {
	root := repoRoot(t)
	fx, err := LoadTokensFixture(filepath.Join(root, "testdata", "golden", "sample_tokens.json"))
	require.NoError(t, err)
	require.NotEmpty(t, fx.Sentences)
	require.NotEmpty(t, fx.Sentences[0].Tokens)
	first := fx.Sentences[0].Tokens[0]
	require.NotEmpty(t, first.Orth)
}

func TestLoadSampleInput(t *testing.T) {
	root := repoRoot(t)
	fx, err := LoadInputFixture(filepath.Join(root, "testdata", "golden", "sample_input.json"))
	require.NoError(t, err)
	require.Len(t, fx.Sentences, 3)
}
