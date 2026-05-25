package registry

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/bioshock/gospacy/v3/nn/layers"
)

func init() {
	// spacy.* current architectures
	Register("spacy.Tagger.v2", buildTaggerV2)
	Register("spacy.TransitionBasedParser.v2", buildTransitionBasedParserV2)
	Register("spacy.TransitionModel.v1", buildTransitionModelV1)
	Register("spacy.PrecomputableAffine.v1", buildPrecomputableAffineV1)
	Register("spacy.Tok2Vec.v2", buildTok2VecV2)
	Register("spacy.Tok2VecListener.v1", buildTok2VecListenerV1)
	Register("spacy.MultiHashEmbed.v2", buildMultiHashEmbedV2)
	Register("spacy.HashEmbedCNN.v2", buildHashEmbedCNNV2)
	Register("spacy.CharacterEmbed.v2", buildCharacterEmbedV2)
	Register("spacy.MaxoutWindowEncoder.v2", buildMaxoutWindowEncoderV2)
	Register("spacy.MishWindowEncoder.v2", buildMishWindowEncoderV2)
	Register("spacy.StaticVectors.v2", buildStaticVectorsV2)
	Register("spacy.FeatureExtractor.v1", buildFeatureExtractorV1)
	Register("spacy.Tokenizer.v1", stub("spacy.Tokenizer.v1", "Phase 3"))

	// spacy-legacy.*
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
		Register(name, stub(name, "Phase 4+ (legacy)"))
	}
}

func stub(name, phase string) ArchitectureFactory {
	return func(cfg map[string]any) (*nn.Model, error) {
		return nil, &ErrArchitectureNotImplemented{Name: name, Phase: phase}
	}
}

func cfgInt(cfg map[string]any, key string, def int) int {
	v, ok := cfg[key]
	if !ok || v == nil {
		return def
	}
	switch x := v.(type) {
	case int64:
		return int(x)
	case float64:
		return int(x)
	case int:
		return x
	}
	return def
}

func cfgBool(cfg map[string]any, key string, def bool) bool {
	v, ok := cfg[key]
	if !ok || v == nil {
		return def
	}
	b, ok := v.(bool)
	if !ok {
		return def
	}
	return b
}

func cfgStringList(cfg map[string]any, key string) ([]string, error) {
	raw, ok := cfg[key]
	if !ok || raw == nil {
		return nil, fmt.Errorf("cfg missing %q", key)
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("cfg[%q] must be []any, got %T", key, raw)
	}
	out := make([]string, len(list))
	for i, v := range list {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("cfg[%q][%d] must be string, got %T", key, i, v)
		}
		out[i] = s
	}
	return out, nil
}

func cfgIntList(cfg map[string]any, key string) ([]int, error) {
	raw, ok := cfg[key]
	if !ok || raw == nil {
		return nil, fmt.Errorf("cfg missing %q", key)
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("cfg[%q] must be []any, got %T", key, raw)
	}
	out := make([]int, len(list))
	for i, v := range list {
		switch x := v.(type) {
		case int64:
			out[i] = int(x)
		case float64:
			out[i] = int(x)
		case int:
			out[i] = x
		default:
			return nil, fmt.Errorf("cfg[%q][%d] must be int, got %T", key, i, v)
		}
	}
	return out, nil
}

// buildHashEmbedCNNV2 implements spacy.HashEmbedCNN.v2.
//
// Architecture: MultiHashEmbed(embed) >> MaxoutWindowEncoder(enc)
//
// MultiHashEmbed is not a named layer in nn/layers; we compose it from
// 4 HashEmbed tables (one per hash seed, matching spaCy's thinc default)
// concatenated, then projected via Linear down to `width`. The encoder
// is MaxoutWindowEncoder repeated `depth` times (each a Residual block of
// Maxout + ExpandWindow).
//
// Signature notes (verified via grep):
//
//	layers.HashEmbed(ops, nO, nV int, seed uint32) *nn.Model
//	layers.Concatenate(ops, sublayers ...*nn.Model) *nn.Model
//	layers.Linear(ops, nO, nI int) *nn.Model
//	layers.Maxout(ops, nO, nI, nP int) *nn.Model
//	layers.ExpandWindow(ops, nW int) *nn.Model
//	layers.Residual(ops, inner *nn.Model) *nn.Model
//	layers.Chain(ops, sublayers ...*nn.Model) *nn.Model
//	gonum.New() *gonum.Ops  (implements nn.Ops)
func buildHashEmbedCNNV2(cfg map[string]any) (*nn.Model, error) {
	width := cfgInt(cfg, "width", 96)
	depth := cfgInt(cfg, "depth", 4)
	embedSize := cfgInt(cfg, "embed_size", 2000)
	windowSize := cfgInt(cfg, "window_size", 1)
	maxoutPieces := cfgInt(cfg, "maxout_pieces", 3)
	// subword_features and pretrained_vectors affect the real spaCy pipeline
	// but are ignored in this structural stub (no training/inference with real data).
	_ = cfgBool(cfg, "subword_features", true)

	if width <= 0 {
		return nil, fmt.Errorf("HashEmbedCNN.v2: width must be > 0, got %d", width)
	}
	if depth <= 0 {
		return nil, fmt.Errorf("HashEmbedCNN.v2: depth must be > 0, got %d", depth)
	}

	ops := gonum.New()

	// Embed stage: 4 HashEmbed tables (seeds 0–3), concatenated → nO=4*width,
	// then projected to width via Linear. This mirrors spaCy's MultiHashEmbed
	// which uses 4 hash functions by default.
	nSeeds := 4
	tables := make([]*nn.Model, nSeeds)
	for i := 0; i < nSeeds; i++ {
		tables[i] = layers.HashEmbed(ops, width, embedSize, uint32(i))
	}
	embed := layers.Chain(ops,
		layers.Concatenate(ops, tables...),
		layers.Linear(ops, width, width*nSeeds),
	)

	// Encoder stage: depth × Residual(Maxout(ExpandWindow(X)))
	// Each block: expand the context window, apply Maxout, add residual.
	// Input width of ExpandWindow: width; output: (2*windowSize+1)*width.
	// Maxout projects back to width so the residual add is shape-compatible.
	encLayers := make([]*nn.Model, depth)
	innerWidth := (2*windowSize + 1) * width
	for d := 0; d < depth; d++ {
		block := layers.Chain(ops,
			layers.ExpandWindow(ops, windowSize),
			layers.Maxout(ops, width, innerWidth, maxoutPieces),
		)
		encLayers[d] = layers.Residual(ops, block)
	}

	enc := layers.Chain(ops, encLayers...)

	model := layers.Chain(ops, embed, enc)
	return model, nil
}

// buildMultiHashEmbedV2 implements spacy.MultiHashEmbed.v2.
//
// Reproduces spacy.ml.models.tok2vec.MultiHashEmbed in both modes
// (include_static_vectors false/true). Returns the COMPLETE embed sub-tree
// (FeatureExtractor → list2ragged → ... → ragged2list), matching upstream's
// factoring where the entire embed pipeline lives inside MultiHashEmbed:
//
// include_static_vectors = false (en_core_web_sm shape, 5-children chain):
//
//	Chain(
//	  FeatureExtractor(attrs),
//	  List2Ragged,
//	  WithArray(Concatenate(IntsGetitem(i) >> HashEmbed(width, rows[i], seed=8+i) × N)),
//	  WithArray(Maxout(width, N*width, nP=3, normalize=True, dropout=0.0)),
//	  Ragged2List,
//	)
//
// include_static_vectors = true (en_core_web_md/lg shape, 3-children chain):
//
//	Chain(
//	  Concatenate(  // Ragged-output dispatch: produces Ragged(Cols=(N+1)*width)
//	    Chain(FeatureExtractor(attrs), List2Ragged, WithArray(Concatenate(...))),
//	    StaticVectors(nO=width, nM=300),  // accepts []*doc.Doc, returns Ragged
//	  ),
//	  WithArray(Maxout(width, (N+1)*width, nP=3, normalize=True, dropout=0.0)),
//	  Ragged2List,
//	)
//
// The StaticVectors arm reads its vector table from
// Attrs["vocab_vectors"] (*vocab.Vectors) and projects via W: (width, 300). The
// bundle loader injects vocab.Vectors after FromBytes since vectors come from
// vocab/vectors, not the tagger/parser model msgpack.
//
// Seeds for hash-embed arms start at 8 because Python's MultiHashEmbed
// initialises `seed = 7` then pre-increments per column. Rows length must
// equal attrs length.
func buildMultiHashEmbedV2(cfg map[string]any) (*nn.Model, error) {
	width := cfgInt(cfg, "width", 96)
	attrsList, err := cfgStringList(cfg, "attrs")
	if err != nil {
		return nil, fmt.Errorf("MultiHashEmbed.v2: %w", err)
	}
	rowsList, err := cfgIntList(cfg, "rows")
	if err != nil {
		return nil, fmt.Errorf("MultiHashEmbed.v2: %w", err)
	}
	if len(attrsList) != len(rowsList) {
		return nil, fmt.Errorf("MultiHashEmbed.v2: len(attrs)=%d != len(rows)=%d",
			len(attrsList), len(rowsList))
	}
	includeStatic := cfgBool(cfg, "include_static_vectors", false)
	maxoutPieces := cfgInt(cfg, "maxout_pieces", 3)

	ops := gonum.New()

	// FeatureExtractor (consumes []*doc.Doc, emits []Uint64s2d).
	colsList := make([]any, len(attrsList))
	for i, s := range attrsList {
		colsList[i] = s
	}
	fe, err := buildFeatureExtractorV1(map[string]any{"columns": colsList})
	if err != nil {
		return nil, fmt.Errorf("MultiHashEmbed.v2: feature extractor: %w", err)
	}

	// Inner hash-embed concatenate: with_array(concat(ints-getitem>>hashembed × N)).
	subs := make([]*nn.Model, len(attrsList))
	for i := range attrsList {
		seed := uint32(8 + i)
		he := layers.HashEmbed(ops, width, rowsList[i], seed)
		he.Attrs["column"] = i
		he.Attrs["dropout_rate"] = float32(0.1)
		subs[i] = layers.Chain(ops,
			layers.IntsGetitem(ops, i),
			he,
		)
	}
	hashEmbedWA := layers.WithArray(ops, layers.Concatenate(ops, subs...))

	// Reduce: with_array(chain(chain(Maxout(nP, normalize), LayerNorm), Dropout)).
	// concatWidth is N*width (sm) or (N+1)*width (md, accounting for the
	// StaticVectors arm contributing one more `width`-wide chunk).
	concatWidth := width * len(attrsList)
	if includeStatic {
		concatWidth += width
	}
	reduceWA := layers.WithArray(ops, layers.Chain(ops,
		layers.Chain(ops,
			layers.Maxout(ops, width, concatWidth, maxoutPieces),
			layers.LayerNorm(ops, width),
		),
		layers.Dropout(ops, 0.0),
	))

	if includeStatic {
		// md/lg shape: feature_extractor sub-chain produces Ragged(Cols=N*width);
		// static_vectors arm produces Ragged(Cols=width); outer concat merges
		// them row-wise into Ragged(Cols=(N+1)*width). Then reduce → ragged2list.
		feChain := layers.Chain(ops, fe, layers.List2Ragged(ops), hashEmbedWA)
		// nM = 300 is the canonical spaCy static-vector dim (md and lg both
		// ship 300-dim vectors). The actual nM is overwritten by FromBytes from
		// the on-disk static_vectors node's Dims["nM"].
		sv := layers.StaticVectors(ops, width, 300)
		outerConcat := layers.Concatenate(ops, feChain, sv)
		embed := layers.Chain(ops, outerConcat, reduceWA, layers.Ragged2List(ops))
		return embed, nil
	}

	// sm shape: flat 5-children chain.
	return layers.Chain(ops,
		fe,
		layers.List2Ragged(ops),
		hashEmbedWA,
		reduceWA,
		layers.Ragged2List(ops),
	), nil
}

// buildMaxoutWindowEncoderV2 implements spacy.MaxoutWindowEncoder.v2.
//
// Reproduces spacy.ml.models.tok2vec.MaxoutWindowEncoder:
//
//	WithArray(Chain(
//	  Residual(Chain(ExpandWindow(w), Maxout(width, (2w+1)*width, nP), LayerNorm(width), Dropout(0.1))),
//	  ... × depth
//	))
//
// All depth blocks share input/output width so the residual add is valid.
func buildMaxoutWindowEncoderV2(cfg map[string]any) (*nn.Model, error) {
	width := cfgInt(cfg, "width", 96)
	depth := cfgInt(cfg, "depth", 4)
	windowSize := cfgInt(cfg, "window_size", 1)
	maxoutPieces := cfgInt(cfg, "maxout_pieces", 3)
	if depth <= 0 {
		return nil, fmt.Errorf("MaxoutWindowEncoder.v2: depth must be > 0, got %d", depth)
	}

	ops := gonum.New()
	innerWidth := (2*windowSize + 1) * width
	residuals := make([]*nn.Model, depth)
	for d := 0; d < depth; d++ {
		residuals[d] = layers.Residual(ops, layers.Chain(ops,
			layers.ExpandWindow(ops, windowSize),
			layers.Maxout(ops, width, innerWidth, maxoutPieces),
			layers.LayerNorm(ops, width),
			layers.Dropout(ops, 0.1),
		))
	}
	return layers.WithArray(ops, layers.Chain(ops, residuals...)), nil
}

// buildMishWindowEncoderV2 implements spacy.MishWindowEncoder.v2.
//
// Reads "width", "depth", "window_size" from cfg.
// Uses Mish activation instead of Maxout.
func buildMishWindowEncoderV2(cfg map[string]any) (*nn.Model, error) {
	width := cfgInt(cfg, "width", 96)
	depth := cfgInt(cfg, "depth", 4)
	windowSize := cfgInt(cfg, "window_size", 1)

	ops := gonum.New()
	innerWidth := (2*windowSize + 1) * width
	encLayers := make([]*nn.Model, depth)
	for d := 0; d < depth; d++ {
		block := layers.Chain(ops,
			layers.ExpandWindow(ops, windowSize),
			layers.Mish(ops, width, innerWidth),
		)
		encLayers[d] = layers.Residual(ops, block)
	}
	model := layers.Chain(ops, encLayers...)
	return model, nil
}

// buildStaticVectorsV2 implements spacy.StaticVectors.v2.
//
// Reads "nO" and "nM" from cfg (matching upstream's spacy.ml.staticvectors.StaticVectors
// dims: nO = output projection width, nM = vector-table column dim). For
// en_core_web_md / _lg this is nO=96, nM=300. Falls back to sensible defaults
// for tests that don't supply the cfg keys.
//
// The vector table itself is injected into the layer's Attrs by the bundle
// loader (see bundle.FromDisk) after FromBytes, because vectors live in
// `vocab/vectors` (not in the tagger/parser model msgpack).
func buildStaticVectorsV2(cfg map[string]any) (*nn.Model, error) {
	nO := cfgInt(cfg, "nO", 96)
	nM := cfgInt(cfg, "nM", 300)
	ops := gonum.New()
	model := layers.StaticVectors(ops, nO, nM)
	return model, nil
}

// buildTok2VecV2 implements spacy.Tok2Vec.v2 for en_core_web_sm / md / lg.
//
// Reproduces spacy.ml.models.tok2vec.build_Tok2Vec_model, which composes:
//
//	chain(embed, encode)
//
// where embed = spacy.MultiHashEmbed.v2 (full embed sub-tree including
// FeatureExtractor through Ragged2List — see buildMultiHashEmbedV2) and
// encode = spacy.MaxoutWindowEncoder.v2 (the with_array(residual chain)).
//
// Walk-order targets (FromBytes parity):
//   - en_core_web_sm: 65 nodes (width=96, 6 columns, depth=4, ws=1, mp=3)
//   - en_core_web_md/lg: 68 nodes (same params + StaticVectors arm: +3 nodes)
//
// Sub-config dispatch: cfg["embed"] / cfg["encode"] (each a map[string]any with
// "@architectures" + per-architecture keys) are routed via registry.Build.
// When absent, the flat cfg is fed to the canonical builders directly. The
// flat-cfg path synthesises canonical attrs/rows and propagates
// include_static_vectors from the top-level cfg so a bundle loader that hands
// us only the flat keys (no nested embed/encode sub-cfgs) still hits the right
// shape.
func buildTok2VecV2(cfg map[string]any) (*nn.Model, error) {
	// Resolve attrs/columns. Prefer cfg["attrs"] (canonical MultiHashEmbed key),
	// fall back to cfg["columns"] (legacy FeatureExtractor key), and finally to
	// the en_core_web_sm default 6-column list when neither is present.
	attrsList, _ := cfgStringList(cfg, "attrs")
	if attrsList == nil {
		attrsList, _ = cfgStringList(cfg, "columns")
	}
	if attrsList == nil {
		attrsList = []string{"NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"}
	}

	// Embed sub-tree: built via MultiHashEmbed.v2 (handles sm + md/lg shapes
	// via include_static_vectors). Dispatch via registry if cfg["embed"] is a
	// nested architecture spec; else fall back to buildMultiHashEmbedV2 with
	// the flat cfg.
	embedCfg := cfg
	if _, has := cfg["attrs"]; !has {
		embedCfg = cloneCfgWithDefaults(cfg, attrsList)
	}
	embed, err := dispatchSubArch(embedCfg, "embed", "spacy.MultiHashEmbed.v2", buildMultiHashEmbedV2)
	if err != nil {
		return nil, fmt.Errorf("Tok2Vec.v2: embed: %w", err)
	}

	// Encode sub-tree: with_array(chain(residual(...) × depth)). Built inline
	// because buildMaxoutWindowEncoderV2 emits a flat cnn without the
	// chain(chain(Maxout, LayerNorm), Dropout) inner wrapping that thinc's
	// Maxout(normalize=True, dropout=0.0) produces.
	ops := gonum.New()
	encodeWA, err := buildTok2VecEncode(cfg, ops)
	if err != nil {
		return nil, fmt.Errorf("Tok2Vec.v2: encode: %w", err)
	}

	return layers.Chain(ops, embed, encodeWA), nil
}

// cloneCfgWithDefaults returns a shallow copy of cfg with synthesised "attrs"
// and "rows" entries (en_core_web_sm canonical 6-column embed) so the embed
// sub-builder can run even when the caller's cfg omitted them. This keeps the
// shape correct for FromBytes to either accept (canonical bundles) or reject
// loudly (drifted bundles) without erroring at Build time.
//
// "include_static_vectors" is propagated as-is so md/lg bundles trigger the
// 3-children embed shape, and sm bundles continue to use the 5-children shape.
func cloneCfgWithDefaults(cfg map[string]any, attrs []string) map[string]any {
	out := make(map[string]any, len(cfg)+2)
	for k, v := range cfg {
		out[k] = v
	}
	if _, ok := out["attrs"]; !ok {
		as := make([]any, len(attrs))
		for i, s := range attrs {
			as[i] = s
		}
		out["attrs"] = as
	}
	if _, ok := out["rows"]; !ok {
		// Canonical en_core_web_sm row counts for the 6-column embed.
		out["rows"] = []any{int64(5000), int64(1000), int64(2500), int64(2500), int64(50), int64(50)}
	}
	return out
}

// dispatchSubArch resolves cfg[key]["@architectures"] via the registry if cfg
// has that nested sub-cfg (typical when the bundle loader passes nested config
// sections), else invokes the fallback builder on the flat cfg (typical when
// the caller hand-builds a cfg map).
func dispatchSubArch(cfg map[string]any, key, defaultArch string, fallback ArchitectureFactory) (*nn.Model, error) {
	if raw, ok := cfg[key]; ok {
		sub, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("cfg[%q] must be map[string]any, got %T", key, raw)
		}
		arch := defaultArch
		if a, ok := sub["@architectures"].(string); ok && a != "" {
			arch = a
		}
		return Build(arch, sub)
	}
	return fallback(cfg)
}

// buildTok2VecEncode builds the with_array(chain(residual(cnn) × depth)) encode
// sub-tree with the thinc-faithful nested cnn = chain(EW, chain(chain(Maxout, LN), Drop)).
//
// Sets `pad = window_size * depth` on the outer with_array, matching
// spacy.ml.models.tok2vec.MaxoutWindowEncoder. The pad rows are zero-padding
// added on both sides of each Doc before flattening, so each Doc's
// ExpandWindow operations see zero rows beyond its own token boundary rather
// than leaking into the neighbouring Doc. The receptive_field is
// window_size*depth because every residual adds one expand_window of
// half-width window_size, and depth such layers stack their receptive fields
// additively.
func buildTok2VecEncode(cfg map[string]any, ops nn.Ops) (*nn.Model, error) {
	width := cfgInt(cfg, "width", 96)
	depth := cfgInt(cfg, "depth", 4)
	windowSize := cfgInt(cfg, "window_size", 1)
	maxoutPieces := cfgInt(cfg, "maxout_pieces", 3)
	if depth <= 0 {
		return nil, fmt.Errorf("Tok2Vec.v2 encode: depth must be > 0, got %d", depth)
	}
	innerWidth := (2*windowSize + 1) * width
	residuals := make([]*nn.Model, depth)
	for d := 0; d < depth; d++ {
		residuals[d] = layers.Residual(ops, layers.Chain(ops,
			layers.ExpandWindow(ops, windowSize),
			layers.Chain(ops,
				layers.Chain(ops,
					layers.Maxout(ops, width, innerWidth, maxoutPieces),
					layers.LayerNorm(ops, width),
				),
				layers.Dropout(ops, 0.0),
			),
		))
	}
	wa := layers.WithArray(ops, layers.Chain(ops, residuals...))
	wa.Attrs["pad"] = windowSize * depth
	return wa, nil
}

// buildTaggerV2 implements spacy.Tagger.v2.
//
// Mirrors spacy/ml/models/tagger.py `build_tagger_model`:
//
//	Chain(tok2vec, with_array(Softmax_v2(nO, t2v_width)))
//
// In an en_core_web_sm bundle, `tok2vec` is a `Tok2VecListener.v1` (a 1-node
// inference proxy that re-uses the upstream tok2vec component's output).
// Hence the on-disk shape is exactly 4 walk-order nodes:
//
//	[0] chain          tok2vec-listener>>with_array(softmax)
//	[1] tok2vec-listener
//	[2] with_array(softmax)
//	[3] softmax
//
// Dim "nO" (label count) and softmax W/b are set by FromBytes from the real
// payload, so the Go-side defaults are only required to build a valid tree;
// the real values land at load time.
//
// Reads "nO" (label count, may be nil/null) and "width" (tok2vec output width)
// from cfg. The width is the listener's `nO` dim and the softmax's `nI` dim.
func buildTaggerV2(cfg map[string]any) (*nn.Model, error) {
	nO := cfgInt(cfg, "nO", 50)
	if nO <= 0 {
		// `nO = null` in config.cfg → cfg["nO"] == int64(0). The real label
		// count is recovered from the on-disk dims via FromBytes; we just need
		// a positive placeholder to build a valid Softmax tree.
		nO = 50
	}
	width := cfgInt(cfg, "width", 96)

	ops := gonum.New()

	// Build the tok2vec listener directly so this factory does not depend on
	// the not-yet-resolved subsection cfg. The listener has no params; its
	// upstream/width attrs get overwritten by FromBytes if present.
	listener := &nn.Model{
		Name:  "tok2vec_listener",
		Ops:   ops,
		Dims:  map[string]int{"nO": width},
		Attrs: map[string]any{"upstream": "tok2vec", "width": int64(width)},
	}

	model := layers.Chain(ops,
		listener,
		layers.WithArray(ops, layers.Softmax(ops, nO, width)),
	)
	return model, nil
}

// buildTransitionBasedParserV2 implements spacy.TransitionBasedParser.v2 (the
// parser's outermost factory). Reads state_type, extra_state_tokens,
// hidden_width, maxout_pieces, use_upper, nO, and the nested tok2vec sub-cfg.
//
// The on-disk shape of the parser model is:
//
//	parser_model
//	├── Chain(Tok2VecListener, List2Array, Linear(width → hidden_width))   refs.tok2vec
//	├── PrecomputableAffine(nO=hidden_width, nI=hidden_width, nF, nP)      refs.lower
//	└── Linear(nO=n_moves, nI=hidden_width)                                refs.upper
//
// FromBytes overwrites Linear.nO with the on-disk n_moves (106 for en_core_web_sm).
func buildTransitionBasedParserV2(cfg map[string]any) (*nn.Model, error) {
	stateType, _ := cfg["state_type"].(string)
	if stateType != "parser" && stateType != "ner" {
		// Phase 7 Block D added NER support; other state_types (e.g. senter
		// or custom transition systems) remain deferred. Surface as
		// ErrArchitectureNotImplemented so the bundle loader records the
		// component as Skipped instead of aborting the entire load.
		return nil, &ErrArchitectureNotImplemented{
			Name:  fmt.Sprintf("spacy.TransitionBasedParser.v2[state_type=%q]", stateType),
			Phase: "Phase 7 supports parser + ner; other state_types deferred",
		}
	}
	extra := cfgBool(cfg, "extra_state_tokens", false)
	nF := 8
	if extra {
		nF = 13
	}
	hiddenWidth := cfgInt(cfg, "hidden_width", 64)
	maxoutPieces := cfgInt(cfg, "maxout_pieces", 2)
	useUpper := cfgBool(cfg, "use_upper", true)
	if !useUpper {
		return nil, fmt.Errorf("TransitionBasedParser.v2: use_upper=false not supported (Phase 5)")
	}

	// tok2vec sub-cfg. The bundle loader's current buildFactoryCfg only feeds
	// the flat top-level keys for a component's model, so the nested
	// [components.parser.model.tok2vec] sub-section may be absent. Fall back
	// to the en_core_web_sm-shaped default listener config when missing —
	// FromBytes will still flag actual on-disk drift via the walk-order check.
	var t2vCfgRaw map[string]any
	if raw, ok := cfg["tok2vec"].(map[string]any); ok {
		t2vCfgRaw = raw
	} else {
		t2vCfgRaw = map[string]any{
			"@architectures": "spacy.Tok2VecListener.v1",
			"width":          int64(96),
			"upstream":       "tok2vec",
		}
	}
	t2vArch, _ := t2vCfgRaw["@architectures"].(string)
	if t2vArch == "" {
		t2vArch = "spacy.Tok2VecListener.v1"
	}
	t2vWidth := cfgInt(t2vCfgRaw, "width", 96)
	listener, err := Build(t2vArch, t2vCfgRaw)
	if err != nil {
		return nil, fmt.Errorf("TransitionBasedParser.v2: tok2vec build: %w", err)
	}

	ops := gonum.New()
	// Wrap the listener in Chain(listener, List2Array, Linear(width → hidden_width)).
	// This matches the on-disk node-1 child shape in en_core_web_sm
	// (listener → list2array → linear projection 96 → 64).
	tok2vec := layers.Chain(ops,
		listener,
		layers.List2Array(ops),
		layers.Linear(ops, hiddenWidth, t2vWidth),
	)

	lower := layers.PrecomputableAffine(ops, hiddenWidth, hiddenWidth, nF, maxoutPieces)

	// upper.nO is recovered at FromBytes time from the on-disk dims block.
	// Placeholder is 1 (positive int); upper's nI is fixed at hiddenWidth.
	upper := layers.Linear(ops, 1, hiddenWidth)

	return buildTransitionModelV1Tree(tok2vec, lower, upper, ops), nil
}

// buildTransitionModelV1 implements spacy.TransitionModel.v1. The on-disk
// parser_model node carries dims (nI) and refs (tok2vec, lower, upper) —
// FromBytes resolves the refs by walking children. At cfg-Build time we
// cannot reconstruct the children from a flat cfg alone, so this factory only
// supports the test path where children are passed via the magic "_layers"
// key (a slice of *nn.Model in tok2vec/lower/upper order). Production callers
// use TransitionBasedParser.v2.
func buildTransitionModelV1(cfg map[string]any) (*nn.Model, error) {
	layersAny, ok := cfg["_layers"].([]*nn.Model)
	if !ok || len(layersAny) != 3 {
		return nil, fmt.Errorf("TransitionModel.v1: this factory only supports the test path with cfg[\"_layers\"] = []*nn.Model{tok2vec, lower, upper}")
	}
	return buildTransitionModelV1Tree(layersAny[0], layersAny[1], layersAny[2], gonum.New()), nil
}

// buildTransitionModelV1Tree composes parser_model with the three children.
// Used by both TransitionBasedParser.v2 and TransitionModel.v1 paths.
func buildTransitionModelV1Tree(tok2vec, lower, upper *nn.Model, ops nn.Ops) *nn.Model {
	m := &nn.Model{
		Name:   "parser_model",
		Ops:    ops,
		Layers: []*nn.Model{tok2vec, lower, upper},
		Refs: map[string]*nn.Model{
			"tok2vec": tok2vec,
			"lower":   lower,
			"upper":   upper,
		},
		Dims:  map[string]int{},
		Attrs: map[string]any{"has_upper": true},
	}
	if v, ok := tok2vec.Dims["nI"]; ok {
		m.Dims["nI"] = v
	}
	return m
}

// buildPrecomputableAffineV1 implements spacy.PrecomputableAffine.v1, the
// parser's `lower` layer. cfg keys: nO, nI, nF, nP. The dropout attr is
// inference-irrelevant; we accept and discard it.
func buildPrecomputableAffineV1(cfg map[string]any) (*nn.Model, error) {
	nO := cfgInt(cfg, "nO", 0)
	nI := cfgInt(cfg, "nI", 0)
	nF := cfgInt(cfg, "nF", 0)
	nP := cfgInt(cfg, "nP", 1)
	if nO <= 0 || nI <= 0 || nF <= 0 || nP <= 0 {
		return nil, fmt.Errorf("PrecomputableAffine.v1: nO/nI/nF/nP must all be positive (got %d/%d/%d/%d)", nO, nI, nF, nP)
	}
	return layers.PrecomputableAffine(gonum.New(), nO, nI, nF, nP), nil
}
