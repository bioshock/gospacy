package config

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestConfig_Simple(t *testing.T) {
	src := `
[nlp]
lang = "en"
pipeline = ["tok2vec","tagger"]

[components.tagger]
factory = "tagger"
threshold = 0.5
n_labels = 50
disabled = false
`
	cfg, err := Parse([]byte(src))
	require.NoError(t, err)

	require.Equal(t, "en", cfg.GetString("nlp.lang"))
	require.Equal(t, []any{"tok2vec", "tagger"}, cfg.GetList("nlp.pipeline"))
	require.Equal(t, "tagger", cfg.GetString("components.tagger.factory"))
	require.InDelta(t, 0.5, cfg.GetFloat("components.tagger.threshold"), 1e-9)
	require.Equal(t, int64(50), cfg.GetInt("components.tagger.n_labels"))
	require.Equal(t, false, cfg.GetBool("components.tagger.disabled"))
}

func TestConfig_ArchitectureRef(t *testing.T) {
	src := `
[model]
@architectures = "spacy.HashEmbedCNN.v2"
width = 96
depth = 4
`
	cfg, err := Parse([]byte(src))
	require.NoError(t, err)
	require.Equal(t, "spacy.HashEmbedCNN.v2", cfg.GetString("model.@architectures"))
	require.Equal(t, int64(96), cfg.GetInt("model.width"))
	require.Equal(t, int64(4), cfg.GetInt("model.depth"))
}

func TestConfig_RealBundleSmoke(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm", "config.cfg")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("model not downloaded: %v", err)
	}
	cfg, err := Parse(data)
	require.NoError(t, err)
	require.Equal(t, "en", cfg.GetString("nlp.lang"))
	pipeline := cfg.GetList("nlp.pipeline")
	require.NotEmpty(t, pipeline)
	require.Contains(t, pipeline, "tagger")
	require.Contains(t, pipeline, "parser")
	require.Contains(t, pipeline, "ner")
}

func TestParse_InterpolationColon(t *testing.T) {
	src := []byte(`[a]
x = 96
[b]
width = ${a:x}
`)
	cfg, err := Parse(src)
	require.NoError(t, err)
	require.Equal(t, int64(96), cfg.GetInt("b.width"))
}

func TestParse_InterpolationDot(t *testing.T) {
	src := []byte(`[nlp]
lang = "en"
[initialize.lookups]
lang = ${nlp.lang}
`)
	cfg, err := Parse(src)
	require.NoError(t, err)
	require.Equal(t, "en", cfg.GetString("initialize.lookups.lang"))
}

func TestParse_InterpolationUnresolved(t *testing.T) {
	src := []byte(`[a]
x = ${b:missing}
`)
	_, err := Parse(src)
	require.Error(t, err)
	require.Contains(t, err.Error(), "${b:missing}")
}

func TestParse_InterpolationNested(t *testing.T) {
	src := []byte(`[paths]
root = "/data"
[paths.sub]
file = ${paths.root}
[use]
where = ${paths.sub:file}
`)
	cfg, err := Parse(src)
	require.NoError(t, err)
	require.Equal(t, "/data", cfg.GetString("use.where"))
}
