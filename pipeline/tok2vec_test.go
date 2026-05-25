package pipeline_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/bioshock/gospacy/v3/bundle"
	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/stretchr/testify/require"
)

// tok2vecGoldenPath returns the absolute path to the golden file produced by
// testharness/dump_tok2vec_per_layer.py.
func tok2vecGoldenPath(t *testing.T) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "testdata", "golden", "tok2vec_per_layer.json")
}

// arrayJSON is the dump_tok2vec_per_layer.py array_to_json shape: shape +
// flattened data. Captured generically per-boundary; the test reshapes per
// kind (2D vs 1D-uint64) on read.
type arrayJSON struct {
	Shape []int           `json:"shape"`
	Dtype string          `json:"dtype"`
	Data  json.RawMessage `json:"data"`
}

type tok2vecGolden struct {
	Text       string `json:"text"`
	NTokens    int    `json:"n_tokens"`
	Boundaries struct {
		ExtractFeatures []arrayJSON `json:"extract_features"`
		List2Ragged     struct {
			Data    arrayJSON `json:"data"`
			Lengths []int32   `json:"lengths"`
		} `json:"list2ragged"`
		MultiHashEmbed struct {
			Data    arrayJSON `json:"data"`
			Lengths []int32   `json:"lengths"`
		} `json:"multi_hash_embed"`
		EmbedReduce struct {
			Data    arrayJSON `json:"data"`
			Lengths []int32   `json:"lengths"`
		} `json:"embed_reduce"`
		Ragged2List []arrayJSON `json:"ragged2list"`
		Encode      []arrayJSON `json:"encode"`
	} `json:"boundaries"`
	Final []arrayJSON `json:"final"`
}

// TestTok2Vec_FullForward_MatchesPythonPerLayer is the Phase 4.5 gold-standard
// parity test: load the real en_core_web_sm bundle's tok2vec pipe, run it on
// the same fixed sentence as dump_tok2vec_per_layer.py, and compare every
// architectural boundary tensor element-by-element against the Python golden.
//
// Tolerances:
//   - 1e-4 for float32 boundaries (loosened from the per-op 1e-5 because
//     accumulated FMA error compounds across 65 nodes).
//   - bit-exact for the uint64 extract_features and list2ragged hash arrays.
//
// What this test asserts (the value Phase 4.5 ships):
//   - the rebuilt Tok2Vec.v2 tree (Task 9–10) loads from disk cleanly via
//     bundle.FromDisk (no Skipped, no FromBytes error)
//   - every sub-layer forward pass matches Python within 1e-4
//   - tokenizer.ToDoc produces the same token count as Python's nlp.make_doc
//     (required because token-count drift would cascade into hash drift)
func TestTok2Vec_FullForward_MatchesPythonPerLayer(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	bundlePath := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present at %s: run testharness/download_assets.sh", bundlePath)
	}
	raw, err := os.ReadFile(tok2vecGoldenPath(t))
	require.NoError(t, err, "run testharness/.venv/bin/python testharness/dump_tok2vec_per_layer.py first")
	var g tok2vecGolden
	require.NoError(t, json.Unmarshal(raw, &g))

	b, err := bundle.FromDisk(bundlePath)
	require.NoError(t, err)
	t2v := b.Pipes["tok2vec"]
	require.NotNil(t, t2v, "tok2vec pipe must be present")
	require.Falsef(t, t2v.Skipped,
		"tok2vec pipe must load cleanly (skip reason: %q)", t2v.SkippedReason)
	require.NotNil(t, t2v.Model)

	d := b.Tokenizer.ToDoc(b.Vocab, g.Text)
	require.Equal(t, g.NTokens, d.NumTokens(),
		"tokenizer must produce same token count as Python (drift would corrupt every downstream hash)")

	// Drive the model layer by layer, asserting parity at each architectural
	// boundary. Walking each sub-layer this way (rather than just gating on
	// the final output) is the gold-standard diagnostic: when a layer
	// diverges, the failure names the layer, the magnitude, and the index.
	//
	// The tok2vec tree's outer Chain has 2 children:
	//   layers[0] = chain(extract_features, list2ragged, with_array(concat),
	//                     with_array(reduce-maxout), ragged2list)
	//   layers[1] = with_array(residual×4)   ← the encoder
	t2vModel := t2v.Model
	require.Len(t, t2vModel.Layers, 2, "outer Tok2Vec chain must have 2 children (embed-chain, encode)")
	embedChain := t2vModel.Layers[0]
	encode := t2vModel.Layers[1]
	require.Len(t, embedChain.Layers, 5,
		"embed chain must have 5 children (extract_features, list2ragged, multi_hash_embed, embed_reduce, ragged2list)")
	extractFeatures := embedChain.Layers[0]
	list2ragged := embedChain.Layers[1]
	multiHashEmbed := embedChain.Layers[2]
	embedReduce := embedChain.Layers[3]
	ragged2list := embedChain.Layers[4]

	// Boundary 1: extract_features (List[Uint64s2d]). Bit-exact: every entry
	// is a hash from doc.Token attrs.
	rawFE, err := extractFeatures.Predict([]any{d})
	require.NoError(t, err)
	feList, ok := rawFE.([]nn.Uint64s2d)
	require.Truef(t, ok, "extract_features must emit []Uint64s2d, got %T", rawFE)
	require.Len(t, feList, len(g.Boundaries.ExtractFeatures))
	for i, want := range g.Boundaries.ExtractFeatures {
		wantU64, err := decodeUint64Data(want.Data)
		require.NoError(t, err)
		require.Equal(t, want.Shape[0], feList[i].Rows, "extract_features[%d].Rows", i)
		require.Equal(t, want.Shape[1], feList[i].Cols, "extract_features[%d].Cols", i)
		assertU64Equal(t, wantU64, feList[i].Data, "extract_features doc %d", i)
	}

	// Boundary 2: list2ragged on Uint64 path → RaggedU64. Bit-exact.
	rawL2R, err := list2ragged.Predict(feList)
	require.NoError(t, err)
	l2r, ok := rawL2R.(nn.RaggedU64)
	require.Truef(t, ok, "list2ragged on uint64 path must emit RaggedU64, got %T", rawL2R)
	wantL2RData, err := decodeUint64Data(g.Boundaries.List2Ragged.Data.Data)
	require.NoError(t, err)
	require.Equal(t, g.Boundaries.List2Ragged.Lengths, l2r.Lengths, "list2ragged.Lengths")
	assertU64Equal(t, wantL2RData, l2r.Data, "list2ragged")

	// Boundary 3: multi_hash_embed → Ragged (float32). Tolerance 1e-4.
	rawMHE, err := multiHashEmbed.Predict(l2r)
	require.NoError(t, err)
	mhe, ok := rawMHE.(nn.Ragged)
	require.Truef(t, ok, "multi_hash_embed must emit Ragged, got %T", rawMHE)
	wantMHEData, err := decodeFloat32Data(g.Boundaries.MultiHashEmbed.Data.Data)
	require.NoError(t, err)
	require.Equal(t, g.Boundaries.MultiHashEmbed.Lengths, mhe.Lengths, "multi_hash_embed.Lengths")
	diff.AssertFloats(t, wantMHEData, mhe.Data, 1e-4, "multi_hash_embed")

	// Boundary 4: embed_reduce → Ragged (width=96). Tolerance 1e-4.
	rawER, err := embedReduce.Predict(mhe)
	require.NoError(t, err)
	er, ok := rawER.(nn.Ragged)
	require.Truef(t, ok, "embed_reduce must emit Ragged, got %T", rawER)
	wantERData, err := decodeFloat32Data(g.Boundaries.EmbedReduce.Data.Data)
	require.NoError(t, err)
	require.Equal(t, g.Boundaries.EmbedReduce.Lengths, er.Lengths, "embed_reduce.Lengths")
	diff.AssertFloats(t, wantERData, er.Data, 1e-4, "embed_reduce")

	// Boundary 5: ragged2list → FloatList. Tolerance 1e-4.
	rawR2L, err := ragged2list.Predict(er)
	require.NoError(t, err)
	r2l, ok := rawR2L.(nn.FloatList)
	require.Truef(t, ok, "ragged2list must emit FloatList, got %T", rawR2L)
	require.Len(t, r2l.Items, len(g.Boundaries.Ragged2List))
	for i, want := range g.Boundaries.Ragged2List {
		wantData, err := decodeFloat32Data(want.Data)
		require.NoError(t, err)
		require.Equal(t, want.Shape[0], r2l.Items[i].Rows, "ragged2list[%d].Rows", i)
		require.Equal(t, want.Shape[1], r2l.Items[i].Cols, "ragged2list[%d].Cols", i)
		diff.AssertFloats(t, wantData, r2l.Items[i].Data, 1e-4, fmt.Sprintf("ragged2list item %d", i))
	}

	// Boundary 6: encode → FloatList. The final tok2vec output.
	rawEnc, err := encode.Predict(r2l)
	require.NoError(t, err)
	enc, ok := rawEnc.(nn.FloatList)
	require.Truef(t, ok, "encode must emit FloatList, got %T", rawEnc)
	require.Len(t, enc.Items, len(g.Boundaries.Encode))
	for i, want := range g.Boundaries.Encode {
		wantData, err := decodeFloat32Data(want.Data)
		require.NoError(t, err)
		require.Equal(t, want.Shape[0], enc.Items[i].Rows, "encode[%d].Rows", i)
		require.Equal(t, want.Shape[1], enc.Items[i].Cols, "encode[%d].Cols", i)
		diff.AssertFloats(t, wantData, enc.Items[i].Data, 1e-4, fmt.Sprintf("encode item %d", i))
	}

	// Final full-forward sanity check (equivalent to model.predict([doc])).
	rawOut, err := t2vModel.Predict([]any{d})
	require.NoError(t, err)
	out, ok := rawOut.(nn.FloatList)
	require.Truef(t, ok, "final output must be FloatList, got %T", rawOut)
	require.Len(t, out.Items, 1)
	require.Len(t, g.Final, 1)
	wantFinalData, err := decodeFloat32Data(g.Final[0].Data)
	require.NoError(t, err)
	diff.AssertFloats(t, wantFinalData, out.Items[0].Data, 1e-4, "tok2vec final output (full forward)")
}

// decodeFloat32Data unmarshals the array_to_json `data` field (a flat JSON
// array of numbers) into a []float32. Held out of the struct so the same
// arrayJSON type can carry uint64 hash data on the extract_features /
// list2ragged boundaries.
func decodeFloat32Data(raw json.RawMessage) ([]float32, error) {
	var fs []float32
	if err := json.Unmarshal(raw, &fs); err != nil {
		return nil, err
	}
	return fs, nil
}

// decodeUint64Data unmarshals the array_to_json `data` field into a []uint64.
// Used for the bit-exact hash-equality assertions on extract_features and
// list2ragged.
func decodeUint64Data(raw json.RawMessage) ([]uint64, error) {
	var us []uint64
	if err := json.Unmarshal(raw, &us); err != nil {
		return nil, err
	}
	return us, nil
}

// assertU64Equal reports the first divergent index for two uint64 slices.
// Used in place of diff.AssertFloats for the hash arrays where any drift
// (not just numerical wobble) is a bug.
func assertU64Equal(t *testing.T, want, got []uint64, format string, args ...any) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf(format+": length mismatch: want %d got %d", append(args, len(want), len(got))...)
	}
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf(format+": first divergence at index %d: want %d got %d",
				append(args, i, want[i], got[i])...)
		}
	}
}
