package registry

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/nn"
)

func TestRegistry_LookupUnknown(t *testing.T) {
	_, err := Build("totally.unknown.v9", nil)
	var unknown *ErrUnknownArchitecture
	require.ErrorAs(t, err, &unknown)
	require.Equal(t, "totally.unknown.v9", unknown.Name)
}

func TestRegistry_LookupRegisteredButStub(t *testing.T) {
	_, err := Build("spacy.Tokenizer.v1", nil)
	var stub *ErrArchitectureNotImplemented
	require.ErrorAs(t, err, &stub)
}

func TestRegistry_BuildHashEmbedCNN(t *testing.T) {
	model, err := Build("spacy.HashEmbedCNN.v2", map[string]any{
		"width":       int64(96),
		"depth":       int64(4),
		"embed_size":  int64(2000),
		"window_size": int64(1),
		"maxout_pieces": int64(3),
		"subword_features": true,
		"pretrained_vectors": nil,
	})
	require.NoError(t, err)
	require.NotNil(t, model)
	_, ok := interface{}(model).(*nn.Model)
	require.True(t, ok)
}

func TestFeatureExtractorV1_Build(t *testing.T) {
	m, err := Build("spacy.FeatureExtractor.v1", map[string]any{
		"columns": []any{"NORM", "PREFIX", "SUFFIX", "SHAPE"},
	})
	require.NoError(t, err)
	require.NotNil(t, m)
	cols, ok := m.Attrs["columns"].([]string)
	require.True(t, ok, "columns must be promoted to []string")
	require.Equal(t, []string{"NORM", "PREFIX", "SUFFIX", "SHAPE"}, cols)
}

func TestTok2VecListenerV1_Build(t *testing.T) {
	m, err := Build("spacy.Tok2VecListener.v1", map[string]any{
		"width":    int64(96),
		"upstream": "tok2vec",
	})
	require.NoError(t, err)
	require.NotNil(t, m)
	require.Equal(t, "tok2vec", m.Attrs["upstream"])
	require.Equal(t, int64(96), m.Attrs["width"])
}

func TestCharacterEmbedV2_Build(t *testing.T) {
	m, err := Build("spacy.CharacterEmbed.v2", map[string]any{
		"nM": int64(64),
		"nC": int64(8),
	})
	require.NoError(t, err)
	require.NotNil(t, m)
	require.Equal(t, "charembed", m.Name)
	// nO must equal nC*nM (per upstream init).
	require.Equal(t, int64(8*64), m.Attrs["nO"])
}

func TestRegistry_LegacyNamespaceRegistered(t *testing.T) {
	for _, name := range []string{
		"spacy-legacy.Tagger.v1",
		"spacy-legacy.TransitionBasedParser.v1",
		"spacy-legacy.Tok2Vec.v1",
		"spacy-legacy.MultiHashEmbed.v1",
		"spacy-legacy.HashEmbedCNN.v1",
		"spacy-legacy.CharacterEmbed.v1",
		"spacy-legacy.MaxoutWindowEncoder.v1",
		"spacy-legacy.MishWindowEncoder.v1",
	} {
		_, err := Build(name, map[string]any{})
		require.NotNil(t, err)
		var unknown *ErrUnknownArchitecture
		require.Falsef(t, errors.As(err, &unknown), "%s must be registered (got ErrUnknownArchitecture)", name)
	}
}
