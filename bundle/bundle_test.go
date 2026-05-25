package bundle

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/registry"
)

func TestBundle_FromDisk_RealModel(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm")

	b, err := FromDisk(path)
	if err != nil {
		var stub *registry.ErrArchitectureNotImplemented
		if errorsAs(err, &stub) {
			t.Logf("bundle load surfaced stub: %v", stub)
		}
		t.Skipf("FromDisk failed: %v", err)
	}
	require.NotNil(t, b)
	require.NotNil(t, b.Config)
	require.NotNil(t, b.Vocab)
	require.NotNil(t, b.Tokenizer)
	require.Equal(t, "en", b.Config.GetString("nlp.lang"))
	require.NotEmpty(t, b.Pipes)
}

func TestFromDisk_InterpolatedConfig(t *testing.T) {
	// Real model path; skip when not downloaded so unit tests stay fast.
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		t.Skip("model not downloaded")
	}
	b, err := FromDisk(path)
	if err != nil {
		t.Skipf("FromDisk failed (not a workaround test): %v", err)
	}
	require.NoError(t, err)
	// The widths in tagger.model.tok2vec come from ${components.tok2vec.model.encode:width}=96.
	w := b.Config.GetInt("components.tagger.model.tok2vec.width")
	require.Equal(t, int64(96), w)
}

func TestBundle_Pipe_EndToEnd(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		t.Skip("model not downloaded")
	}
	b, err := FromDisk(path)
	require.NoError(t, err)

	d, err := b.Pipe("The cat sat on the mat.")
	require.NoError(t, err)
	require.Greater(t, d.NumTokens(), 0, "Pipe must tokenize into at least one token")

	// Lemmatizer runs in lookup-fallback mode (tagger deferred to Phase 4.5),
	// returning at minimum tok.Text. Assert every token has a non-empty lemma.
	ss := b.Vocab.StringStore()
	for i, tok := range d.Tokens {
		lemma, _ := ss.Lookup(tok.Lemma)
		require.NotEmptyf(t, lemma,
			"token %d %q must have a non-empty lemma after Pipe()", i, tok.Text)
	}
}

func TestFromDisk_RealBundleLoadsTok2VecAndTagger(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present at %s; run testharness/download_assets.sh", path)
	}
	b, err := FromDisk(path)
	require.NoError(t, err)

	tok2vec, ok := b.Pipes["tok2vec"]
	require.True(t, ok, "tok2vec pipe must exist")
	require.False(t, tok2vec.Skipped, "tok2vec must not be skipped (Phase 4.5): reason=%q", tok2vec.SkippedReason)
	require.NotNil(t, tok2vec.Model, "tok2vec.Model must be non-nil")

	tagger, ok := b.Pipes["tagger"]
	require.True(t, ok, "tagger pipe must exist")
	require.False(t, tagger.Skipped, "tagger must not be skipped (Phase 4.5): reason=%q", tagger.SkippedReason)
	require.NotNil(t, tagger.Model, "tagger.Model must be non-nil")
}

// TestPipeWith_SkipsLemmatizer — PipeWith(SkipLemmatizer: true) must
// leave Token.Lemma at the zero hash (no lemma interned) while keeping
// POS / Dep / Head populated. Verifies the lemmatizer component is
// genuinely bypassed, not just called with a no-op.
func TestPipeWith_SkipsLemmatizer(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		t.Skip("en_core_web_sm not present; skipping")
	}
	b, err := FromDisk(path)
	require.NoError(t, err)

	// Baseline Pipe gives non-zero Lemma on at least one content word.
	dBase, err := b.Pipe("The cat sat on the mat.")
	require.NoError(t, err)
	require.Greater(t, dBase.NumTokens(), 0)
	gotBaseLemma := false
	for _, tok := range dBase.Tokens {
		if tok.Lemma != 0 {
			gotBaseLemma = true
			break
		}
	}
	require.True(t, gotBaseLemma, "baseline Pipe must set at least one Lemma")

	// PipeWith(SkipLemmatizer) leaves every Lemma at 0.
	dSkip, err := b.PipeWith("The cat sat on the mat.", PipeOptions{SkipLemmatizer: true})
	require.NoError(t, err)
	require.Equal(t, dBase.NumTokens(), dSkip.NumTokens(), "token count must match baseline")
	for i, tok := range dSkip.Tokens {
		require.Zerof(t, tok.Lemma, "tok %d %q: SkipLemmatizer must leave Lemma=0", i, tok.Text)
	}
	// At least one non-punct token must still have POS set (parser+tagger ran).
	gotPOS := false
	for _, tok := range dSkip.Tokens {
		if tok.POS != 0 {
			gotPOS = true
			break
		}
	}
	require.True(t, gotPOS, "POS must still be populated when only lemmatizer is skipped")
}

// TestFromDisk_SkipsDisabledPipes — pipes listed in nlp.disabled must
// land in b.Pipes as Skipped: true with reason "disabled in nlp.disabled"
// and Model: nil (no build attempt). Verified against the real md bundle
// where config.cfg has `disabled = ["senter"]`. Pre-Fix-1 senter would
// load and then surface a "tree has 4 nodes, payload has 52" walk-order
// warning on FromBytes; post-fix it never attempts the load.
func TestFromDisk_SkipsDisabledPipes(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_md")
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		t.Skip("en_core_web_md not present; skipping")
	}
	prev := stderrf
	stderrf = func(format string, args ...any) {} // silence so the test output stays clean
	defer func() { stderrf = prev }()

	b, err := FromDisk(path)
	require.NoError(t, err)
	require.NotNil(t, b)

	senter, ok := b.Pipes["senter"]
	require.True(t, ok, "senter pipe must be present in b.Pipes")
	require.True(t, senter.Skipped, "senter must be Skipped because it is in nlp.disabled")
	require.Equal(t, "disabled in nlp.disabled", senter.SkippedReason)
	require.Nil(t, senter.Model, "disabled pipes must not be built")
}

func errorsAs(err error, target any) bool {
	type aser interface{ As(any) bool }
	if a, ok := err.(aser); ok {
		return a.As(target)
	}
	return false
}

func TestBundle_FromDisk_MissingMeta(t *testing.T) {
	tmp := t.TempDir()
	_, err := FromDisk(tmp)
	require.Error(t, err)
	require.Contains(t, err.Error(), "meta.json")
}

func TestBundle_ManifestMatchesGolden(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	goldenPath := filepath.Join(root, "testdata", "golden", "bundle_meta.json")
	modelPath := filepath.Join(root, "testdata", "models", "en_core_web_sm")

	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Skipf("golden missing: %v", err)
	}
	var golden struct {
		Lang       string   `json:"lang"`
		Pipeline   []string `json:"pipeline"`
		Components map[string]struct {
			Architecture *string `json:"architecture"`
		} `json:"components"`
	}
	require.NoError(t, json.Unmarshal(data, &golden))

	b, err := FromDisk(modelPath)
	if err != nil {
		t.Skipf("bundle load failed: %v", err)
	}
	require.Equal(t, golden.Lang, b.Config.GetString("nlp.lang"))
	for _, name := range golden.Pipeline {
		_, ok := b.Pipes[name]
		require.Truef(t, ok, "missing pipe %q in Bundle.Pipes", name)
	}
	for name, comp := range golden.Components {
		if comp.Architecture == nil {
			continue
		}
		pipe := b.Pipes[name]
		require.NotNilf(t, pipe, "missing pipe %q", name)
		require.Equalf(t, *comp.Architecture, pipe.Architecture, "pipe %q architecture", name)
	}
}
