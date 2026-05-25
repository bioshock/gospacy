package diff

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// goldenPath returns the absolute path to a checked-in golden fixture.
func goldenPath(t *testing.T, name string) string {
	t.Helper()
	return filepath.Join(repoRoot(t), "testdata", "golden", name)
}

func TestSmoke_TokensRoundTrip(t *testing.T) {
	fx, err := LoadTokensFixture(goldenPath(t, "sample_tokens.json"))
	require.NoError(t, err)
	for _, s := range fx.Sentences {
		// Comparing the fixture to itself should always be equal.
		rep := CompareTokens(s.Tokens, s.Tokens)
		require.True(t, rep.Equal(), "self-comparison should be equal for sentence: %q", s.Text)
		require.Equal(t, ClassEqual, Classify(rep).Primary)
	}
}

func TestSmoke_AttrsRoundTrip(t *testing.T) {
	fx, err := LoadAttrsFixture(goldenPath(t, "sample_attrs.json"))
	require.NoError(t, err)
	require.NotEmpty(t, fx.Pipeline)
	for _, s := range fx.Sentences {
		require.True(t, CompareAttrs(s.Tokens, s.Tokens).Equal())
	}
}

func TestSmoke_ArcsRoundTrip(t *testing.T) {
	fx, err := LoadArcsFixture(goldenPath(t, "sample_arcs.json"))
	require.NoError(t, err)
	for _, s := range fx.Sentences {
		r := CompareArcs(s.Arcs, s.Arcs)
		require.True(t, r.Equal())
		if len(s.Arcs) > 0 {
			require.Equal(t, 1.0, r.LAS)
			require.Equal(t, 1.0, r.UAS)
		}
	}
}

func TestSmoke_EntitiesRoundTrip(t *testing.T) {
	fx, err := LoadEntitiesFixture(goldenPath(t, "sample_entities.json"))
	require.NoError(t, err)
	for _, s := range fx.Sentences {
		r := CompareEntities(s.Entities, s.Entities)
		require.True(t, r.Equal())
		if len(s.Entities) > 0 {
			require.Equal(t, 1.0, r.F1)
		}
	}
}

// Sanity: the pinned spaCy version recorded in the attrs fixture matches UPSTREAM.
func TestSmoke_SpacyVersionMatchesUpstream(t *testing.T) {
	fx, err := LoadAttrsFixture(goldenPath(t, "sample_attrs.json"))
	require.NoError(t, err)
	require.Equal(t, "3.8.14", fx.SpacyVersion,
		"if this fails, regenerate fixtures (`make diff-test`) after updating spaCy pin")
}
