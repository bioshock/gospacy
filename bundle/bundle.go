// Package bundle reads a .spacy model directory from disk: meta.json,
// config.cfg, vocab/, tokenizer, and per-pipe subdirectories. Mirrors
// spacy.util.from_disk.
package bundle

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/bioshock/gospacy/v3/config"
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/lang/en"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/pipeline"
	en_pipeline "github.com/bioshock/gospacy/v3/pipeline/lang/en"
	"github.com/bioshock/gospacy/v3/registry"
	"github.com/bioshock/gospacy/v3/tokenizer"
	"github.com/bioshock/gospacy/v3/vocab"
)

// stderrf is the logging sink for bundle warnings. Routed through a
// package-level var so tests can silence it without touching os.Stderr.
var stderrf = func(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format, args...)
}

// Bundle is the in-memory representation of a loaded .spacy directory.
type Bundle struct {
	Path      string
	Config    *config.Config
	Vocab     *vocab.Vocab
	Tokenizer *tokenizer.Tokenizer
	Pipes     map[string]*Pipe

	tagger         *pipeline.Tagger
	parser         *pipeline.Parser
	attributeRuler *pipeline.AttributeRuler
	lemmatizer     *pipeline.Lemmatizer
	ner            *pipeline.NER
	componentsOnce sync.Once
	componentsErr  error
}

// Pipe is one pipeline component: the architecture name, the instantiated
// model (weights loaded), and a Skipped flag set when the architecture is
// registered but not implemented.
type Pipe struct {
	Name          string
	Architecture  string
	Model         *nn.Model
	Skipped       bool
	SkippedReason string
}

// BundlePath returns the on-disk directory the bundle was loaded from.
// Implements pipeline.BundleSource so pipeline components can locate per-pipe
// data files (e.g. tagger/cfg, parser/moves).
func (b *Bundle) BundlePath() string { return b.Path }

// BundleVocab returns the bundle's Vocab (StringStore + Lexeme cache + vector
// matrix). Implements pipeline.BundleSource so components share one Vocab.
func (b *Bundle) BundleVocab() *vocab.Vocab { return b.Vocab }

// BundleConfig returns the parsed config.cfg. Implements pipeline.BundleSource
// so components can look up their own hyperparameters (e.g. lemmatizer mode).
func (b *Bundle) BundleConfig() *config.Config { return b.Config }

// PipeLookup implements pipeline.BundleSource. Returns (model, skipped,
// skipReason, present) for the named pipe. model is nil when the pipe was
// skipped or the architecture is a stub.
func (b *Bundle) PipeLookup(name string) (model *nn.Model, skipped bool, skipReason string, ok bool) {
	p, exists := b.Pipes[name]
	if !exists {
		return nil, false, "", false
	}
	return p.Model, p.Skipped, p.SkippedReason, true
}

// FromDisk reads a .spacy bundle from path. Supports en_core_web_sm,
// en_core_web_md, and en_core_web_lg (Phase 7 Block C). The bundle's
// vocab/vectors and key2row are loaded into vocab.Vocab; if the bundle's
// tok2vec carries a static_vectors arm (md/lg), the populated *vocab.Vectors
// is injected into the layer's Attrs after FromBytes so the per-token lookup
// in the StaticVectors Forward sees real data.
//
// Pipes that do not match the registered architecture (e.g. NER's
// TransitionBasedParser.v2 with state_type="ner", or senter), or whose
// FromBytes walk-order mismatches the on-disk thinc payload, land as
// Skipped: true with the failure recorded in SkippedReason. Bundle.Pipe
// skips Skipped pipes silently. Aborting the entire bundle on one
// unsupported pipe is too strict for a port that's intentionally
// inference-only with a documented out-of-scope list.
func FromDisk(path string) (*Bundle, error) {
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		return nil, fmt.Errorf("FromDisk: meta.json missing: %w", err)
	}
	cfgBytes, err := os.ReadFile(filepath.Join(path, "config.cfg"))
	if err != nil {
		return nil, fmt.Errorf("FromDisk: read config.cfg: %w", err)
	}
	cfg, err := config.Parse(cfgBytes)
	if err != nil {
		return nil, fmt.Errorf("FromDisk: parse config.cfg: %w", err)
	}

	v := vocab.NewVocab()
	if data, err := os.ReadFile(filepath.Join(path, "vocab", "strings.json")); err == nil {
		if err := v.StringStore().LoadJSON(data); err != nil {
			return nil, fmt.Errorf("FromDisk: load vocab/strings.json: %w", err)
		}
	}
	// Load the populated vector matrix + key2row index (md / lg) or the
	// empty-shape pair (sm). The StaticVectors layer reads from this via
	// Attrs["vocab_vectors"]; we inject after FromBytes for each pipe below.
	vec, err := vocab.LoadVectorsDir(filepath.Join(path, "vocab"))
	if err != nil {
		return nil, fmt.Errorf("FromDisk: load vocab/vectors: %w", err)
	}
	v.SetVectors(vec)

	lang := cfg.GetString("nlp.lang")
	if lang != "en" {
		return nil, fmt.Errorf("FromDisk: only nlp.lang=en supported in v1, got %q", lang)
	}
	rules, err := en.MakeRules()
	if err != nil {
		return nil, fmt.Errorf("FromDisk: build tokenizer rules: %w", err)
	}
	tk := tokenizer.New(rules)

	pipeline := cfg.GetList("nlp.pipeline")
	disabled := disabledSet(cfg)
	pipes := make(map[string]*Pipe, len(pipeline))
	for _, p := range pipeline {
		name, ok := p.(string)
		if !ok {
			return nil, fmt.Errorf("FromDisk: nlp.pipeline element %v: not a string", p)
		}
		// Pipes listed in nlp.disabled are recorded as Skipped without any
		// build or FromBytes attempt. spaCy's loader treats nlp.disabled as
		// "load but do not run"; gospacy treats it as "do not load either" —
		// we don't expose a runtime enable() that would justify spending the
		// build cost on a pipe Bundle.Pipe will never invoke. The empty
		// architecture-mismatch warning for senter on md/lg (Tagger.v2 with
		// embedded full Tok2Vec.v2 vs gospacy's listener-only stub) is
		// silenced by this skip.
		arch := cfg.GetString(fmt.Sprintf("components.%s.model.@architectures", name))
		if _, dis := disabled[name]; dis {
			pipes[name] = &Pipe{
				Name:          name,
				Architecture:  arch, // populated for introspection / manifest golden parity
				Skipped:       true,
				SkippedReason: "disabled in nlp.disabled",
			}
			continue
		}
		if arch == "" {
			pipes[name] = &Pipe{Name: name, Skipped: true, SkippedReason: "no model section"}
			continue
		}
		factoryCfg := buildFactoryCfg(cfg, fmt.Sprintf("components.%s.model", name))
		model, err := registry.Build(arch, factoryCfg)
		if err != nil {
			var stub *registry.ErrArchitectureNotImplemented
			if errors.As(err, &stub) {
				pipes[name] = &Pipe{
					Name:          name,
					Architecture:  arch,
					Skipped:       true,
					SkippedReason: err.Error(),
				}
				continue
			}
			return nil, fmt.Errorf("FromDisk: build %s: %w", name, err)
		}
		modelBytes, err := os.ReadFile(filepath.Join(path, name, "model"))
		if err != nil {
			pipes[name] = &Pipe{
				Name:          name,
				Architecture:  arch,
				Model:         model,
				Skipped:       true,
				SkippedReason: fmt.Sprintf("model file missing: %v", err),
			}
			continue
		}
		if err := model.FromBytes(modelBytes); err != nil {
			// A pipe whose Go tree shape does not match the on-disk thinc
			// payload (e.g. a stub architecture whose tree shape diverges
			// from the bundle) is recorded as Skipped with the FromBytes
			// error rather than aborting the entire bundle. The warning is
			// surfaced via stderrf for visibility.
			stderrf("bundle: pipe %q FromBytes failed: %v (Skipped)\n", name, err)
			pipes[name] = &Pipe{
				Name:          name,
				Architecture:  arch,
				Model:         model,
				Skipped:       true,
				SkippedReason: err.Error(),
			}
			continue
		}
		// Inject the Vocab's *Vectors into every static_vectors node in this
		// pipe's tree. md/lg have a static_vectors arm in tok2vec; sm has none
		// (so this is a no-op on sm). This must happen AFTER FromBytes since
		// FromBytes overwrites Attrs only for keys present in the payload, and
		// "vocab_vectors" is not in the thinc payload (it lives in vocab/vectors).
		if vec != nil && vec.Rows() > 0 {
			for _, node := range model.Walk() {
				if node.Name == "static_vectors" {
					if node.Attrs == nil {
						node.Attrs = map[string]any{}
					}
					node.Attrs["vocab_vectors"] = vec
				}
			}
		}
		pipes[name] = &Pipe{Name: name, Architecture: arch, Model: model}
	}

	return &Bundle{
		Path:      path,
		Config:    cfg,
		Vocab:     v,
		Tokenizer: tk,
		Pipes:     pipes,
	}, nil
}

// Clone returns a deep copy of the Bundle suitable for running Pipe from a
// separate goroutine in parallel with the source.
//
// Concurrency contract — the source bundle must be quiescent during Clone.
// Clone deep-copies the source's Vocab and StringStore maps by ranging over
// them; Pipe lazy-writes both maps on previously-unseen tokens (Vocab.Get
// inserts a fresh Lexeme; StringStore.Add interns the orth/lower/shape/etc.
// strings). Go's map semantics make a concurrent range + write a data race
// regardless of key overlap. So Clone is NOT safe to call while another
// goroutine is calling Pipe on the source — you must either build all
// clones before launching workers, or quiesce the source before cloning.
// The intended pattern is two-phase:
//
//	src, _ := bundle.FromDisk(path)            // ~625 ms once
//	workers := make([]*bundle.Bundle, N)
//	workers[0] = src
//	for i := 1; i < N; i++ {                   // ~114 ms each on en_core_web_md
//	    workers[i], _ = src.Clone()
//	}
//	// now launch one goroutine per workers[i]; no mutex needed
//
// Per Bundle.Pipe's single-goroutine constraint (Vocab/StringStore/
// Lemmatizer.posCache lazy writes + per-layer scratch slices captured in
// Forward closures), the only way to parallelise Pipe is one Bundle per
// goroutine. Clone gives you that without re-reading the on-disk model: it
// skips the disk IO and msgpack decode (~600 ms on en_core_web_md) and only
// re-instantiates the layer closures (so each clone has its own scratch)
// plus deep-copies the mutable vocab + lexeme state. Param slices are
// shared by reference (immutable post-FromBytes — only Forward reads them).
// Tokenizer rules and the parsed Config are also shared (both immutable).
//
// Cost: dominated by registry.Build re-running per pipe (typically tens of
// ms on en_core_web_md vs ~700 ms for FromDisk).
//
// Lazy pipeline components (tagger / parser / AR / lemmatizer / ner) are
// NOT pre-initialised; the first Pipe call on the clone will materialise
// them via ensureComponents, the same as a freshly-loaded Bundle.
func (b *Bundle) Clone() (*Bundle, error) {
	clonedVocab := b.Vocab.Clone()
	clonedPipes := make(map[string]*Pipe, len(b.Pipes))

	pipelineList := b.Config.GetList("nlp.pipeline")
	for _, p := range pipelineList {
		name, ok := p.(string)
		if !ok {
			return nil, fmt.Errorf("Clone: nlp.pipeline element %v: not a string", p)
		}
		srcPipe, exists := b.Pipes[name]
		if !exists {
			continue
		}
		// Skipped pipe: clone the metadata only (no model tree to rebuild).
		if srcPipe.Skipped || srcPipe.Model == nil {
			clonedPipes[name] = &Pipe{
				Name:          srcPipe.Name,
				Architecture:  srcPipe.Architecture,
				Skipped:       srcPipe.Skipped,
				SkippedReason: srcPipe.SkippedReason,
			}
			continue
		}
		// Re-invoke the architecture factory to get fresh closures (so the
		// new clone's layer scratch slices are independent from the source).
		factoryCfg := buildFactoryCfg(b.Config, fmt.Sprintf("components.%s.model", name))
		newModel, err := registry.Build(srcPipe.Architecture, factoryCfg)
		if err != nil {
			return nil, fmt.Errorf("Clone: rebuild %s (%s): %w", name, srcPipe.Architecture, err)
		}
		if err := cloneModelParams(srcPipe.Model, newModel); err != nil {
			return nil, fmt.Errorf("Clone: copy params for %s: %w", name, err)
		}
		// Re-inject the (shared, immutable) Vectors into static_vectors nodes,
		// mirroring FromDisk's post-FromBytes step.
		if vec := clonedVocab.Vectors(); vec != nil && vec.Rows() > 0 {
			for _, node := range newModel.Walk() {
				if node.Name == "static_vectors" {
					if node.Attrs == nil {
						node.Attrs = map[string]any{}
					}
					node.Attrs["vocab_vectors"] = vec
				}
			}
		}
		clonedPipes[name] = &Pipe{
			Name:         srcPipe.Name,
			Architecture: srcPipe.Architecture,
			Model:        newModel,
		}
	}

	return &Bundle{
		Path:      b.Path,
		Config:    b.Config,    // immutable, shared by reference
		Vocab:     clonedVocab, // deep-copied
		Tokenizer: b.Tokenizer, // immutable (holds only *Rules), shared
		Pipes:     clonedPipes,
		// componentsOnce zero-value → first Clone.Pipe lazy-inits its own
		// tagger / parser / AR / lemmatizer / NER (with their own posCache).
	}, nil
}

// cloneModelParams walks src and dst in BFS order (same as FromBytes) and
// copies each node's Params map + Dims map from src to dst. Param []float32
// slices are aliased by reference (Params are read-only after load). Attrs
// are not copied — the dst already has factory-default Attrs (column, seed,
// nW, etc.) and dynamic Attrs like "vocab_vectors" are re-injected by the
// caller above.
func cloneModelParams(src, dst *nn.Model) error {
	srcNodes := src.Walk()
	dstNodes := dst.Walk()
	if len(srcNodes) != len(dstNodes) {
		return fmt.Errorf("walk length mismatch: src=%d dst=%d", len(srcNodes), len(dstNodes))
	}
	for i, sn := range srcNodes {
		dn := dstNodes[i]
		// Do NOT compare names — FromBytes overwrites factory-default names
		// with the on-disk thinc payload's BFS name array (e.g. underscore
		// vs dash, "softmax" vs "softmax_v2"). The structural invariant we
		// rely on is BFS-order match, enforced by the length check above.
		if sn.Params != nil {
			dn.Params = make(map[string][]float32, len(sn.Params))
			for k, v := range sn.Params {
				dn.Params[k] = v // share underlying array; Params are read-only post-load
			}
		}
		if sn.Dims != nil {
			dn.Dims = make(map[string]int, len(sn.Dims))
			for k, v := range sn.Dims {
				dn.Dims[k] = v
			}
		}
	}
	return nil
}

// PipeOptions selectively disables pipeline components on a single
// PipeWith call. Zero value runs the full pipeline (same as Pipe).
// Mirrors spaCy's `nlp.pipe(..., disable=[...])` and
// `with nlp.select_pipes(disable=[...])` semantics — additive, no
// global state. Useful for parser-heavy consumers that never read
// Lemma / EntIOB / EntType (skipping lemmatizer + NER on md cuts a
// substantial fraction of per-Pipe latency since md ships NER active).
//
// Ordering caveat: AttributeRuler runs BEFORE Lemmatizer in the
// pipeline; the lemmatizer reads POS to choose lookup mode. Setting
// SkipAttributeRuler=true while SkipLemmatizer=false will change the
// lemmas the lemmatizer produces (it sees the un-AR-corrected POS).
// For most parser-heavy consumers SkipLemmatizer=true is the simpler
// pairing.
type PipeOptions struct {
	SkipAttributeRuler bool
	SkipLemmatizer     bool
	SkipNER            bool
}

// Pipe runs the full annotation pipeline on text: tokenize → tok2vec →
// tagger → parser → attribute_ruler → lemmatizer → ner. Equivalent to
// PipeWith(text, PipeOptions{}). Components are constructed lazily on
// first call and cached.
//
// Pipe is NOT safe for concurrent invocation on a shared *Bundle. Every
// call mutates per-Bundle state on previously-unseen inputs: Vocab.Get
// lazily interns new lexemes into Vocab.lexemes and StringStore.keys2str,
// and Lemmatizer.Apply lazily fills per-POS lookup caches. These maps are
// unsynchronized; concurrent callers will hit `fatal error: concurrent map
// writes` or silent corruption. To process texts in parallel, construct
// one *Bundle per goroutine. See vocab.Vocab.Get, vocab.StringStore.Add,
// and pipeline.Lemmatizer.posCache for the underlying constraints. The
// constraint is locked down by bundle/bundle_race_test.go (build tag
// `raceverify`, see OVERVIEW.md §11).
func (b *Bundle) Pipe(text string) (*doc.Doc, error) {
	return b.PipeWith(text, PipeOptions{})
}

// PipeWith is Pipe with selective component skipping. See PipeOptions.
func (b *Bundle) PipeWith(text string, opts PipeOptions) (*doc.Doc, error) {
	if err := b.ensureComponents(); err != nil {
		return nil, err
	}
	d := b.Tokenizer.ToDoc(b.Vocab, text)
	var tok2vecOut nn.Floats2d
	needTok2vec := b.tagger != nil || b.parser != nil
	if needTok2vec {
		tok2vecPipe, ok := b.Pipes["tok2vec"]
		if !ok || tok2vecPipe == nil || tok2vecPipe.Skipped || tok2vecPipe.Model == nil {
			return nil, fmt.Errorf("Bundle.Pipe: tagger/parser requires loaded tok2vec, found nil/skipped")
		}
		rawOut, err := tok2vecPipe.Model.Predict([]any{d})
		if err != nil {
			return nil, fmt.Errorf("Bundle.Pipe: tok2vec: %w", err)
		}
		list, ok := rawOut.(nn.FloatList)
		if !ok || len(list.Items) != 1 {
			return nil, fmt.Errorf("Bundle.Pipe: tok2vec returned %T (want FloatList of len 1)", rawOut)
		}
		tok2vecOut = list.Items[0]
	}
	if b.tagger != nil {
		if err := b.tagger.Apply(d, tok2vecOut); err != nil {
			return nil, fmt.Errorf("Bundle.Pipe: tagger: %w", err)
		}
	}
	if b.parser != nil {
		if err := b.parser.Apply(d, tok2vecOut); err != nil {
			return nil, fmt.Errorf("Bundle.Pipe: parser: %w", err)
		}
	}
	if b.attributeRuler != nil && !opts.SkipAttributeRuler {
		if err := b.attributeRuler.Apply(d); err != nil {
			return nil, fmt.Errorf("Bundle.Pipe: attribute_ruler: %w", err)
		}
	}
	if b.lemmatizer != nil && !opts.SkipLemmatizer {
		if err := b.lemmatizer.Apply(d); err != nil {
			return nil, fmt.Errorf("Bundle.Pipe: lemmatizer: %w", err)
		}
	}
	if b.ner != nil && !opts.SkipNER {
		// NER uses its own non-listener Tok2Vec.v2 — run that chain
		// separately. The chain (Refs["tok2vec"]) is
		// Tok2Vec.v2 → list2array → linear; its output is the projected
		// T × hidden_width Floats2d that NER.Apply feeds into lower.
		nerPipe := b.Pipes["ner"]
		if nerPipe == nil || nerPipe.Model == nil {
			return nil, fmt.Errorf("Bundle.Pipe: ner is configured but Pipes[\"ner\"] is missing")
		}
		nerTok2VecChain := nerPipe.Model.Refs["tok2vec"]
		if nerTok2VecChain == nil {
			return nil, fmt.Errorf("Bundle.Pipe: ner model has no tok2vec ref")
		}
		projectedRaw, err := nerTok2VecChain.Predict([]any{d})
		if err != nil {
			return nil, fmt.Errorf("Bundle.Pipe: ner tok2vec: %w", err)
		}
		projected, ok := projectedRaw.(nn.Floats2d)
		if !ok {
			return nil, fmt.Errorf("Bundle.Pipe: ner tok2vec returned %T (want Floats2d)", projectedRaw)
		}
		if err := b.ner.Apply(d, projected); err != nil {
			return nil, fmt.Errorf("Bundle.Pipe: ner: %w", err)
		}
	}
	return d, nil
}

func (b *Bundle) ensureComponents() error {
	b.componentsOnce.Do(func() {
		tg, err := pipeline.NewTagger(b)
		if err != nil {
			// Tagger skipped (pipe absent or model failed to load) — not fatal.
			stderrf("bundle: ensureComponents: tagger skipped: %v\n", err)
		} else {
			b.tagger = tg
		}
		ps, err := pipeline.NewParser(b)
		if err != nil {
			stderrf("bundle: ensureComponents: parser skipped: %v\n", err)
		} else {
			b.parser = ps
		}
		ar, err := pipeline.NewAttributeRuler(b)
		if err != nil {
			stderrf("bundle: ensureComponents: attribute_ruler skipped: %v\n", err)
		} else {
			b.attributeRuler = ar
		}
		lm, err := pipeline.NewLemmatizer(b)
		if err != nil {
			b.componentsErr = fmt.Errorf("ensureComponents: lemmatizer: %w", err)
			return
		}
		lm.IsBaseForm = func(tok *doc.Token, pos string) bool {
			return en_pipeline.IsBaseForm(tok, pos)
		}
		b.lemmatizer = lm

		ner, err := pipeline.NewNER(b)
		if err != nil {
			// NER absent or Skipped — not fatal. Bundles without NER (or a
			// load failure on a stub bundle) just leave doc.Tokens[].EntIOB
			// at 0 / EntType at 0.
			stderrf("bundle: ensureComponents: ner skipped: %v\n", err)
		} else {
			b.ner = ner
		}
	})
	return b.componentsErr
}

// buildFactoryCfg materialises the cfg sub-tree at sectionPrefix as a nested
// map[string]any. Immediate leaf keys become string→value pairs; nested
// sub-sections become string→map[string]any with the same recursion. This
// matches spaCy's resolution path where each architecture sub-cfg
// ([components.X.model.embed], [components.X.model.encode], etc.) is passed
// to the architecture's factory through a nested dict.
//
// Phase 7 Block C made this nested: md/lg bundles carry
// [components.tok2vec.model.embed].include_static_vectors=true which must
// propagate through Tok2Vec.v2 into MultiHashEmbed.v2 to select the 3-children
// embed shape. Pre-C the flat-key extraction silently dropped any dotted-key
// (sub-section) data and md tok2vec FromBytes failed walk-order mismatch.
func buildFactoryCfg(cfg *config.Config, sectionPrefix string) map[string]any {
	out := map[string]any{}
	pre := sectionPrefix + "."
	prefixLen := len(pre)

	// Collect immediate (no-dot) keys first, then group dotted keys by their
	// first segment for recursive descent.
	subSections := map[string]struct{}{}
	for _, k := range cfg.Subkeys(sectionPrefix) {
		short := k[prefixLen:]
		dotIdx := strings.IndexByte(short, '.')
		if dotIdx < 0 {
			out[short] = readAny(cfg, k)
			continue
		}
		subSections[short[:dotIdx]] = struct{}{}
	}
	for sub := range subSections {
		out[sub] = buildFactoryCfg(cfg, sectionPrefix+"."+sub)
	}
	return out
}

// disabledSet returns the set of pipe names listed under nlp.disabled. Empty
// when the key is absent or empty.
func disabledSet(cfg *config.Config) map[string]struct{} {
	out := map[string]struct{}{}
	for _, e := range cfg.GetList("nlp.disabled") {
		if name, ok := e.(string); ok {
			out[name] = struct{}{}
		}
	}
	return out
}

func readAny(cfg *config.Config, path string) any {
	if cfg.Has(path) {
		if s := cfg.GetString(path); s != "" {
			return s
		}
		if l := cfg.GetList(path); l != nil {
			return l
		}
		if cfg.GetBool(path) {
			return true
		}
		if v := cfg.GetInt(path); v != 0 {
			return v
		}
		if v := cfg.GetFloat(path); v != 0 {
			return v
		}
		return int64(0)
	}
	return nil
}
