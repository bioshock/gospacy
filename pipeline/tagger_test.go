package pipeline_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/bioshock/gospacy/v3/nn/layers"
	"github.com/bioshock/gospacy/v3/pipeline"
	"github.com/bioshock/gospacy/v3/vocab"
)

// TestTagger_Apply_SyntheticWeights is a deterministic unit test that proves
// the argmax→Tag wiring works in isolation, with no dependency on the real
// en_core_web_sm bundle.
//
// Construction: 1 token, 3 labels, width=2. The softmax bias steers argmax
// without needing matrix-multiply contributions, so the expected tag is
// known regardless of W. Hand-set weights are:
//
//	W = [[1, 0], [0, 1], [0, 0]]   shape (nO=3, nI=2)
//	b = [-10, 5, -10]              shape (3,)
//
// With tok2vecOutput = [[0, 0]] the row scores are b = [-10, 5, -10]; the
// argmax is index 1 → label "VERB". This isolates the wiring (matrix-vector
// affine + softmax + argmax + StringStore.Add → Token.Tag) from any dependency
// on the model's parameter values being meaningful.
func TestTagger_Apply_SyntheticWeights(t *testing.T) {
	ops := gonum.New()

	// Mirror the real 4-node tagger tree shape: Chain(listener, with_array(softmax)).
	listener := &nn.Model{
		Name:  "tok2vec_listener",
		Ops:   ops,
		Dims:  map[string]int{"nO": 2},
		Attrs: map[string]any{"upstream": "tok2vec", "width": int64(2)},
	}
	softmax := layers.Softmax(ops, 3, 2)
	softmax.Params["W"] = []float32{1, 0, 0, 1, 0, 0}
	softmax.Params["b"] = []float32{-10, 5, -10}

	model := layers.Chain(ops, listener, layers.WithArray(ops, softmax))
	require.Len(t, model.Walk(), 4, "synthetic tagger tree must mirror real-bundle 4-node shape")

	v := vocab.NewVocab()
	b := &bundle.Bundle{Vocab: v}
	tg := pipeline.NewTaggerFromModel(model, []string{"NOUN", "VERB", "ADJ"}, b)

	d := doc.NewDoc(v, "hi")
	d.Tokens = []doc.Token{{Text: "hi"}}

	tok2vecOutput := nn.Floats2d{Data: []float32{0, 0}, Rows: 1, Cols: 2}
	require.NoError(t, tg.Apply(d, tok2vecOutput))

	// argmax of bias [-10, 5, -10] is index 1 → "VERB".
	wantHash, ok := v.StringStore().Get("VERB")
	require.True(t, ok, "VERB must have been interned")
	require.Equal(t, wantHash, d.Tokens[0].Tag,
		"argmax→Tag must resolve to VERB hash")
}

// TestTagger_Apply_MultipleTokens verifies the per-row argmax also writes
// distinct tags when the softmax bias varies per row via the W contribution
// from different feature rows. Two tokens, identical W, but different inputs
// → different scores → different tags. Proves the Ragged unwrap loop iterates
// every token row, not just the first.
func TestTagger_Apply_MultipleTokens(t *testing.T) {
	ops := gonum.New()

	listener := &nn.Model{
		Name:  "tok2vec_listener",
		Ops:   ops,
		Dims:  map[string]int{"nO": 2},
		Attrs: map[string]any{"upstream": "tok2vec", "width": int64(2)},
	}
	// W rows in thinc convention are (nO, nI). Each row selects one feature:
	//   row0 = [10, 0] → label 0 wins when input feature 0 dominates
	//   row1 = [0, 10] → label 1 wins when input feature 1 dominates
	softmax := layers.Softmax(ops, 2, 2)
	softmax.Params["W"] = []float32{10, 0, 0, 10}
	softmax.Params["b"] = []float32{0, 0}

	model := layers.Chain(ops, listener, layers.WithArray(ops, softmax))

	v := vocab.NewVocab()
	b := &bundle.Bundle{Vocab: v}
	tg := pipeline.NewTaggerFromModel(model, []string{"NOUN", "VERB"}, b)

	d := doc.NewDoc(v, "ab")
	d.Tokens = []doc.Token{{Text: "a"}, {Text: "b"}}

	// Row 0 features [1, 0] → score [10, 0] → argmax = NOUN.
	// Row 1 features [0, 1] → score [0, 10] → argmax = VERB.
	tok2vecOutput := nn.Floats2d{
		Data: []float32{1, 0, 0, 1},
		Rows: 2,
		Cols: 2,
	}
	require.NoError(t, tg.Apply(d, tok2vecOutput))

	nounHash, ok := v.StringStore().Get("NOUN")
	require.True(t, ok)
	verbHash, ok := v.StringStore().Get("VERB")
	require.True(t, ok)
	require.Equal(t, nounHash, d.Tokens[0].Tag, "row 0 must tag as NOUN")
	require.Equal(t, verbHash, d.Tokens[1].Tag, "row 1 must tag as VERB")
}

// TestTagger_RealBundle_DifferentialPenn loads the real en_core_web_sm tagger
// pipe to verify Phase 4's bundle/buildTaggerV2 changes load the on-disk
// weights cleanly (pipe.Skipped == false, pipe.Model != nil), then SKIPs
// before running the differential comparison against testdata/golden/tagger.json
// because the upstream tok2vec layer parity is deferred to Phase 4.5 per the
// 2026-05-19 plan amendment.
//
// What this test asserts (the value Phase 4 ships):
//   - the reframed buildTaggerV2 builds a tree whose walk-order matches the
//     real tagger payload (4 nodes), so FromBytes succeeds
//   - bundle.FromDisk tolerates the unloadable tok2vec pipe (Skipped) and
//     keeps going to load the tagger pipe successfully
//   - NewTagger can be constructed from the loaded pipe and tagger/cfg
//
// What this test does NOT yet assert (Phase 4.5):
//   - the actual POS argmax matches Python on the 8 fixture sentences
//     (requires the tok2vec forward pass, which depends on the missing
//     thinc layers: LayerNorm, Dropout, IntsGetitem, ExtractFeatures,
//     Ragged2List, MultiHashEmbed concat, HashEmbed with nI, residual+
//     expand_window wiring)
func TestTagger_RealBundle_DifferentialPenn(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	modelPath := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(modelPath, "meta.json")); err != nil {
		t.Skip("model not downloaded")
	}

	b, err := bundle.FromDisk(modelPath)
	require.NoError(t, err, "bundle.FromDisk must tolerate Phase 3 stub mismatches and load tagger cleanly")

	pipe, ok := b.Pipes["tagger"]
	require.True(t, ok, "real bundle must expose a 'tagger' pipe")
	require.Falsef(t, pipe.Skipped,
		"tagger pipe must load cleanly with the reframed buildTaggerV2 (skipped reason: %q)",
		pipe.SkippedReason)
	require.NotNil(t, pipe.Model, "loaded tagger pipe must have a non-nil model")

	tg, err := pipeline.NewTagger(b)
	require.NoError(t, err, "NewTagger must accept the loaded tagger pipe")
	require.NotNil(t, tg)
	// Phase 4.5 (Task 13) added the end-to-end differential in
	// TestTagger_RealBundle_ProducesPythonMatchingTags. This test retains
	// its narrow Phase-4 assertions (pipe loads cleanly + NewTagger
	// constructs) as a fast smoke check that the real-bundle tagger pipe
	// is still loadable before the differential runs.
}
