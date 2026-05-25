package registry_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/registry"
)

func TestMultiHashEmbedV2_BuildsSixColumnTree(t *testing.T) {
	cfg := map[string]any{
		"width": int64(96),
		"attrs": []any{"NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"},
		"rows":  []any{int64(5000), int64(1000), int64(2500), int64(2500), int64(50), int64(50)},
		"include_static_vectors": false,
	}
	m, err := registry.Build("spacy.MultiHashEmbed.v2", cfg)
	require.NoError(t, err)

	// Post-Phase-7-Block-C, buildMultiHashEmbedV2 returns the COMPLETE embed
	// sub-tree (FeatureExtractor → list2ragged → with_array(concat6) →
	// with_array(reduce) → ragged2list) — 30 nodes for sm shape. This matches
	// upstream's MultiHashEmbed Python builder factoring and is the prerequisite
	// for the include_static_vectors=true variant (which restructures into a
	// 3-children chain).
	//
	// BFS layout:
	//   [0]    chain (5 children: fe, l2r, wa-concat6, wa-reduce, r2l)
	//   [1]    extract_features
	//   [2]    list2ragged
	//   [3]    with_array(concat6)
	//   [4]    with_array(reduce)
	//   [5]    ragged2list
	//   [6]    concat6 (inside wa-concat6)
	//   [7]    chain(chain(maxout,layernorm), dropout)  -- reduce body
	//   [8..13] 6 chain(ints-getitem, hashembed) children of concat6
	//   [14]   chain(maxout, layernorm)
	//   [15]   dropout
	//   [16..27] 12× (ints-getitem, hashembed) tuples (6 pairs)
	//   [28]   maxout
	//   [29]   layernorm
	walked := m.Walk()
	require.Equal(t, 30, len(walked), "walk-order length (full embed sub-tree)")
	require.Equal(t, "extract_features", walked[1].Name)
	require.Equal(t, "list2ragged", walked[2].Name)
	require.Equal(t, "ragged2list", walked[5].Name)
	// Per-column nV and seed on the 6 hashembed leaves (BFS positions 17, 19, ..., 27).
	wantNV := []int{5000, 1000, 2500, 2500, 50, 50}
	wantSeed := []uint32{8, 9, 10, 11, 12, 13}
	for i := 0; i < 6; i++ {
		hashembed := walked[17+i*2]
		require.Equal(t, "hashembed", hashembed.Name, "column %d hashembed slot", i)
		require.Equal(t, wantNV[i], hashembed.Dims["nV"], "column %d nV", i)
		require.Equal(t, wantSeed[i], hashembed.Attrs["seed"].(uint32), "column %d seed", i)
		require.Equal(t, i, hashembed.Attrs["column"].(int), "column %d column-attr", i)
	}
}

// TestMultiHashEmbedV2_BuildsMDShapeWithStaticVectors verifies the md/lg shape
// (include_static_vectors=true) produces a 33-node embed sub-tree matching the
// upstream en_core_web_md tok2vec/model:
//
//   - Top chain has 3 children (concat, with_array(reduce), ragged2list)
//   - Inner concat has 2 children: feature_extractor_chain and static_vectors
//   - The static_vectors leaf has W: (nO, nM) param shape
//
// Without this test md bundles silently fall back to a wrong sub-shape and
// FromBytes flags a walk-order mismatch at load time — but the assertion here
// catches the structural drift at build time.
func TestMultiHashEmbedV2_BuildsMDShapeWithStaticVectors(t *testing.T) {
	cfg := map[string]any{
		"width":                  int64(96),
		"attrs":                  []any{"NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"},
		"rows":                   []any{int64(5000), int64(1000), int64(2500), int64(2500), int64(50), int64(50)},
		"include_static_vectors": true,
	}
	m, err := registry.Build("spacy.MultiHashEmbed.v2", cfg)
	require.NoError(t, err)
	walked := m.Walk()
	// md embed sub-tree: 33 nodes (sm's 30 + 3 for static_vectors + outer
	// concat + fe-sub-chain restructure). Verify the static_vectors leaf and
	// the outer concat wiring.
	require.Equal(t, 33, len(walked), "md embed walk-order length")
	// Position 0 = outer embed Chain with 3 children.
	require.Len(t, m.Layers, 3, "embed chain has 3 children (outer concat, reduce-WA, ragged2list)")
	// Position 1 = outer concat with 2 children (fe-chain, static_vectors).
	outerConcat := walked[1]
	require.Contains(t, outerConcat.Name, "|", "outer concat (Concatenate name joined by '|')")
	require.Len(t, outerConcat.Layers, 2, "outer concat has fe-chain + static_vectors")
	// Find the static_vectors node — it must exist exactly once with the
	// canonical Dims and the W param scaffold.
	found := false
	for _, node := range walked {
		if node.Name == "static_vectors" {
			require.False(t, found, "expected exactly one static_vectors node")
			found = true
			require.Equal(t, 96, node.Dims["nO"])
			require.Equal(t, 300, node.Dims["nM"])
			require.Contains(t, node.Params, "W")
		}
	}
	require.True(t, found, "expected a static_vectors node in the md embed tree")
}

func TestMaxoutWindowEncoderV2_BuildsDepth4Residual(t *testing.T) {
	cfg := map[string]any{
		"width":         int64(96),
		"depth":         int64(4),
		"window_size":   int64(1),
		"maxout_pieces": int64(3),
	}
	m, err := registry.Build("spacy.MaxoutWindowEncoder.v2", cfg)
	require.NoError(t, err)
	walked := m.Walk()
	// with_array + outer chain + 4*(residual + inner chain + expand_window + maxout + layernorm + dropout)
	// = 2 + 4*6 = 26.
	require.Equal(t, 26, len(walked))
	require.Contains(t, walked[0].Name, "with_array")
	// walked[1] is the outer Chain wrapping the 4 residual siblings inside the with_array.
	// Chain layer names are children joined by ">>", so the outer chain name contains "residual".
	require.Contains(t, walked[1].Name, "residual", "outer chain joins residual siblings")

	// BFS layout (matches thinc):
	//   [0]    with_array
	//   [1]    outer chain (4 residual siblings)
	//   [2..5] 4× Residual nodes
	//   [6..9] 4× inner Chain nodes (children of each Residual)
	//   [10..25] 4×(EW, Maxout, LN, Dropout) tuples, one tuple per inner chain.
	for d := 0; d < 4; d++ {
		require.Contains(t, walked[2+d].Name, "residual", "depth %d residual", d)
		require.Contains(t, walked[6+d].Name, ">>", "depth %d inner chain", d)

		ewIdx := 10 + d*4
		require.Equal(t, "expand_window", walked[ewIdx].Name, "depth %d expand_window", d)
		require.Equal(t, "maxout", walked[ewIdx+1].Name, "depth %d maxout", d)
		require.Equal(t, 96, walked[ewIdx+1].Dims["nO"])
		require.Equal(t, 288, walked[ewIdx+1].Dims["nI"])
		require.Equal(t, 3, walked[ewIdx+1].Dims["nP"])
		require.Equal(t, "layernorm", walked[ewIdx+2].Name, "depth %d layernorm", d)
		require.Equal(t, "dropout", walked[ewIdx+3].Name, "depth %d dropout", d)
	}
}

func TestTransitionBasedParserV2_Build(t *testing.T) {
	m, err := registry.Build("spacy.TransitionBasedParser.v2", map[string]any{
		"state_type":         "parser",
		"extra_state_tokens": false,
		"hidden_width":       int64(64),
		"maxout_pieces":      int64(2),
		"use_upper":          true,
		"nO":                 nil,
		"tok2vec": map[string]any{
			"@architectures": "spacy.Tok2VecListener.v1",
			"width":          int64(96),
			"upstream":       "tok2vec",
		},
	})
	require.NoError(t, err)
	require.Equal(t, "parser_model", m.Name)
	require.Len(t, m.Layers, 3, "parser_model has three children: tok2vec, lower, upper")
	require.NotNil(t, m.Refs["tok2vec"])
	require.NotNil(t, m.Refs["lower"])
	require.NotNil(t, m.Refs["upper"])
	require.Equal(t, "precomputable_affine", m.Refs["lower"].Name)
	require.Equal(t, 8, m.Refs["lower"].Dims["nF"])
	require.Equal(t, 64, m.Refs["lower"].Dims["nO"])
	require.Equal(t, 2, m.Refs["lower"].Dims["nP"])
	require.Equal(t, "linear", m.Refs["upper"].Name)
	// upper.nO is a placeholder until FromBytes overwrites from on-disk dims.
	require.Equal(t, 64, m.Refs["upper"].Dims["nI"])

	// Walk: parser_model, tok2vec_chain, lower, upper, listener, list2array, projection_linear
	walk := m.Walk()
	require.Equal(t, 7, len(walk), "walk should be 7 nodes")
	require.Equal(t, "parser_model", walk[0].Name)
	// walk[1] is the tok2vec Chain with name joined by ">>"
	require.Contains(t, walk[1].Name, ">>")
	require.Equal(t, "precomputable_affine", walk[2].Name)
	require.Equal(t, "linear", walk[3].Name)
	require.Equal(t, "tok2vec_listener", walk[4].Name)
	require.Equal(t, "list2array", walk[5].Name)
	require.Equal(t, "linear", walk[6].Name)
}

func TestTransitionBasedParserV2_StateTypeNER(t *testing.T) {
	// Phase 7 Block D: state_type="ner" must build (the same model shape
	// applies — only the TransitionSystem differs at runtime). The NER pipe
	// configures its own non-listener Tok2Vec.v2 sub-cfg; the factory
	// dispatcher routes via @architectures so this works without any
	// NER-specific branching in buildTransitionBasedParserV2.
	m, err := registry.Build("spacy.TransitionBasedParser.v2", map[string]any{
		"state_type":         "ner",
		"extra_state_tokens": false,
		"hidden_width":       int64(64),
		"maxout_pieces":      int64(2),
		"use_upper":          true,
		"nO":                 nil,
		"tok2vec": map[string]any{
			"@architectures": "spacy.Tok2VecListener.v1",
			"width":          int64(96),
			"upstream":       "tok2vec",
		},
	})
	require.NoError(t, err)
	require.Equal(t, "parser_model", m.Name)
	require.NotNil(t, m.Refs["lower"])
	require.NotNil(t, m.Refs["upper"])
	require.Equal(t, 8, m.Refs["lower"].Dims["nF"])
	require.Equal(t, 64, m.Refs["lower"].Dims["nO"])
}

func TestTransitionBasedParserV2_RejectsOtherStateType(t *testing.T) {
	_, err := registry.Build("spacy.TransitionBasedParser.v2", map[string]any{
		"state_type":         "tagger",
		"extra_state_tokens": false,
		"hidden_width":       int64(64),
		"maxout_pieces":      int64(2),
		"use_upper":          true,
	})
	require.Error(t, err)
	var stub *registry.ErrArchitectureNotImplemented
	require.ErrorAs(t, err, &stub, "non-parser/ner state_type must surface as ErrArchitectureNotImplemented")
}

func TestPrecomputableAffineV1_Build(t *testing.T) {
	m, err := registry.Build("spacy.PrecomputableAffine.v1", map[string]any{
		"nO": int64(64),
		"nI": int64(64),
		"nF": int64(8),
		"nP": int64(2),
	})
	require.NoError(t, err)
	require.Equal(t, "precomputable_affine", m.Name)
	require.Equal(t, 8, m.Dims["nF"])
	require.Equal(t, 64, m.Dims["nI"])
	require.Equal(t, 64, m.Dims["nO"])
	require.Equal(t, 2, m.Dims["nP"])
}

func TestTok2VecV2_Builds65NodeTreeForEnCoreWebSm(t *testing.T) {
	cfg := map[string]any{
		"width": int64(96),
		// MultiHashEmbed sub-cfg
		"attrs": []any{"NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"},
		"rows":  []any{int64(5000), int64(1000), int64(2500), int64(2500), int64(50), int64(50)},
		"include_static_vectors": false,
		// MaxoutWindowEncoder sub-cfg
		"depth":         int64(4),
		"window_size":   int64(1),
		"maxout_pieces": int64(3),
		// FeatureExtractor sub-cfg: same column list as attrs.
		"columns": []any{"NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"},
	}
	m, err := registry.Build("spacy.Tok2Vec.v2", cfg)
	require.NoError(t, err)

	walked := m.Walk()
	require.Equal(t, 65, len(walked), "tok2vec walk-order length must equal en_core_web_sm bundle (65 nodes)")
	// gospacy's Walk() is BFS (matches thinc's default `Model.walk()` order),
	// so leaf positions below correspond to the en_core_web_sm bundle's on-disk
	// `nodes[]` array directly. The from_bytes_walkorder_test.go assertion
	// checks the full 65-name sequence; this test pins a handful of structural
	// anchors so a future drift surfaces early.
	require.Equal(t, "extract_features", walked[3].Name)
	require.Equal(t, "list2ragged", walked[4].Name)
	require.Equal(t, "ragged2list", walked[7].Name)
	// 6 hashembed leaves at BFS indices 28, 30, 32, 34, 36, 38 (spacing 2 due
	// to the `chain(ints_getitem, hashembed)` wrapping around each table).
	for _, idx := range []int{28, 30, 32, 34, 36, 38} {
		require.Equal(t, "hashembed", walked[idx].Name, "node %d should be hashembed", idx)
	}
}
