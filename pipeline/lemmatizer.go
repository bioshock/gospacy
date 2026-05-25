package pipeline

import (
	"fmt"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/internal/lookups"
	"github.com/bioshock/gospacy/v3/vocab"
)

// LookupSource is the minimal interface the Lemmatizer needs from a Lookups
// container. *lookups.Lookups satisfies it; tests can supply a fake.
type LookupSource interface {
	Has(name string) bool
	Get(name string) *lookups.Table
}

// Lemmatizer writes Token.Lemma using either a flat lookup table or a
// rule-based pipeline. Mirrors spacy.pipeline.lemmatizer.Lemmatizer for the
// two modes en_core_web_sm and shipping en lemma data exercise: "lookup" and
// "rule".
//
// IsBaseForm is the per-language hook. For English, Bundle.ensureComponents
// installs pipeline/lang/en.IsBaseForm, which mirrors
// spacy.lang.en.lemmatizer.EnglishLemmatizer.is_base_form.
type Lemmatizer struct {
	mode       string
	source     LookupSource
	vocab      *vocab.Vocab
	overwrite  bool
	IsBaseForm func(tok *doc.Token, pos string) bool

	// posCache memoises the per-POS rules/excs/index materialisations so
	// ruleLemmatize doesn't allocate fresh maps and slices on every token.
	// Phase 7 Block B identified readExc/readIndex's per-token copies as the
	// pipeline hot path: ~44% cumulative CPU in runtime.mapassign_faststr
	// alone, building maps of thousands of entries just to look up one key.
	// Cache is lazy-filled the first time a POS is seen; subsequent tokens
	// with the same POS hit the cached *posCache directly. Single-goroutine
	// access only — matches Bundle.Pipe's existing threading model (see
	// bundle.go's lack of per-Apply locking).
	posCache map[uint64]*posCache
}

// posCache holds the materialised lookup payloads for one Universal Dependency
// POS hash. All three fields are read-only once built and shared across every
// token of that POS for the lifetime of the Lemmatizer.
type posCache struct {
	rules    [][2]string
	excs     map[string][]string
	indexSet map[string]struct{}
}

// NewLemmatizer builds the base lemmatizer. Reads:
//   - cfg path components.lemmatizer.mode for "lookup" or "rule" (defaults to "lookup")
//   - <bundle>/lemmatizer/lookups/lookups.bin for the lemma_* tables
func NewLemmatizer(b BundleSource) (*Lemmatizer, error) {
	mode := b.BundleConfig().GetString("components.lemmatizer.mode")
	if mode == "" {
		mode = "lookup"
	}
	overwrite := b.BundleConfig().GetBool("components.lemmatizer.overwrite")
	tablesPath := filepath.Join(b.BundlePath(), "lemmatizer", "lookups", "lookups.bin")
	l, err := lookups.Load(tablesPath)
	if err != nil {
		return nil, fmt.Errorf("NewLemmatizer: load %s: %w", tablesPath, err)
	}
	return &Lemmatizer{mode: mode, source: l, vocab: b.BundleVocab(), overwrite: overwrite}, nil
}

// NewLemmatizerForTest constructs a Lemmatizer from an explicit LookupSource
// (used by unit tests that want to inject a tiny fake without writing to
// disk). Production callers use NewLemmatizer.
func NewLemmatizerForTest(v *vocab.Vocab, src LookupSource, mode string) (*Lemmatizer, error) {
	return &Lemmatizer{mode: mode, source: src, vocab: v}, nil
}

// Apply writes Token.Lemma for every token in d.
func (lm *Lemmatizer) Apply(d *doc.Doc) error {
	for i := range d.Tokens {
		tok := &d.Tokens[i]
		if tok.Lemma != 0 && !lm.overwrite {
			continue
		}
		lemma := lm.lemmatize(tok, d)
		if lemma == "" {
			lemma = tok.Text
		}
		tok.Lemma = d.Vocab.StringStore().Add(lemma)
	}
	return nil
}

func (lm *Lemmatizer) lemmatize(tok *doc.Token, d *doc.Doc) string {
	switch lm.mode {
	case "lookup":
		return lm.lookupLemmatize(tok)
	case "rule":
		return lm.ruleLemmatize(tok, d)
	}
	return tok.Text
}

func (lm *Lemmatizer) lookupLemmatize(tok *doc.Token) string {
	if lm.source == nil || !lm.source.Has("lemma_lookup") {
		return strings.ToLower(tok.Text)
	}
	table := lm.source.Get("lemma_lookup")
	if v, ok := table.GetByHash(tok.Orth); ok {
		switch x := v.(type) {
		case string:
			return x
		case uint64:
			s, _ := lm.vocab.StringStore().Lookup(x)
			return s
		case []any:
			if len(x) > 0 {
				if s, ok := x[0].(string); ok {
					return s
				}
			}
		}
	}
	return tok.Text
}

func (lm *Lemmatizer) ruleLemmatize(tok *doc.Token, d *doc.Doc) string {
	pos, _ := d.Vocab.StringStore().Lookup(tok.POS)
	univ := strings.ToLower(pos)
	if univ == "" || univ == "eol" || univ == "space" {
		return strings.ToLower(tok.Text)
	}
	if lm.IsBaseForm != nil && lm.IsBaseForm(tok, univ) {
		return strings.ToLower(tok.Text)
	}
	// lemma_* tables are keyed by the murmur hash of the lowercased POS string
	// (e.g. "verb", "noun"). Use Hash (not Get): the rule/exc/index payloads were
	// loaded by msgpack and do not carry the string side, so the string has
	// never been Added to the StringStore — Get would return (0, false) and
	// every per-POS lookup would miss. Hash always produces the canonical
	// table key. Mirrors spacy.pipeline.lemmatizer.rule_lemmatize, which keys
	// off univ_pos (a Python string) directly.
	posHash := d.Vocab.StringStore().Hash(univ)

	pc := lm.getPosCache(posHash)
	if len(pc.rules) == 0 && len(pc.excs) == 0 && len(pc.indexSet) == 0 {
		if univ == "propn" {
			return tok.Text
		}
		return strings.ToLower(tok.Text)
	}
	s := strings.ToLower(tok.Text)
	var forms []string
	var oov []string
	for _, pair := range pc.rules {
		old, replacement := pair[0], pair[1]
		if !strings.HasSuffix(s, old) {
			continue
		}
		form := s[:len(s)-len(old)] + replacement
		if form == "" {
			continue
		}
		if _, in := pc.indexSet[form]; in || !isAlpha(form) {
			if _, in := pc.indexSet[form]; in {
				forms = append([]string{form}, forms...)
			} else {
				forms = append(forms, form)
			}
		} else {
			oov = append(oov, form)
		}
	}
	forms = dedupe(forms)
	if got, ok := pc.excs[s]; ok {
		for _, e := range got {
			if !contains(forms, e) {
				forms = append([]string{e}, forms...)
			}
		}
	}
	if len(forms) == 0 {
		forms = oov
	}
	if len(forms) == 0 {
		return tok.Text
	}
	return forms[0]
}

// getPosCache returns the cached posCache for posHash, materialising it from
// the underlying lookups on first call. Always returns a non-nil *posCache so
// the caller can use len() on its fields without nil checks; missing tables
// yield empty slices/maps inside. Lazy: caches grow as new POS values are
// encountered across calls.
func (lm *Lemmatizer) getPosCache(posHash uint64) *posCache {
	if lm.posCache == nil {
		lm.posCache = map[uint64]*posCache{}
	}
	if pc, ok := lm.posCache[posHash]; ok {
		return pc
	}
	index := lm.readIndex(posHash)
	indexSet := make(map[string]struct{}, len(index))
	for _, f := range index {
		indexSet[f] = struct{}{}
	}
	pc := &posCache{
		rules:    lm.readRules(posHash),
		excs:     lm.readExc(posHash),
		indexSet: indexSet,
	}
	lm.posCache[posHash] = pc
	return pc
}

func (lm *Lemmatizer) readRules(pos uint64) [][2]string {
	if lm.source == nil || !lm.source.Has("lemma_rules") {
		return nil
	}
	v, ok := lm.source.Get("lemma_rules").GetByHash(pos)
	if !ok {
		return nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([][2]string, 0, len(list))
	for _, e := range list {
		pair, ok := e.([]any)
		if !ok || len(pair) != 2 {
			continue
		}
		a, _ := pair[0].(string)
		b, _ := pair[1].(string)
		out = append(out, [2]string{a, b})
	}
	return out
}

func (lm *Lemmatizer) readExc(pos uint64) map[string][]string {
	if lm.source == nil || !lm.source.Has("lemma_exc") {
		return nil
	}
	v, ok := lm.source.Get("lemma_exc").GetByHash(pos)
	if !ok {
		return nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		// map[any]any fallback
		if mm, ok2 := v.(map[any]any); ok2 {
			out := make(map[string][]string, len(mm))
			for k, vv := range mm {
				ks, _ := k.(string)
				lst, _ := vv.([]any)
				vals := make([]string, 0, len(lst))
				for _, l := range lst {
					if s, ok := l.(string); ok {
						vals = append(vals, s)
					}
				}
				out[ks] = vals
			}
			return out
		}
		return nil
	}
	out := make(map[string][]string, len(m))
	for k, vv := range m {
		lst, _ := vv.([]any)
		vals := make([]string, 0, len(lst))
		for _, l := range lst {
			if s, ok := l.(string); ok {
				vals = append(vals, s)
			}
		}
		out[k] = vals
	}
	return out
}

func (lm *Lemmatizer) readIndex(pos uint64) []string {
	if lm.source == nil || !lm.source.Has("lemma_index") {
		return nil
	}
	v, ok := lm.source.Get("lemma_index").GetByHash(pos)
	if !ok {
		return nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(list))
	for _, e := range list {
		if s, ok := e.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func dedupe(xs []string) []string {
	seen := map[string]struct{}{}
	out := xs[:0]
	for _, x := range xs {
		if _, ok := seen[x]; ok {
			continue
		}
		seen[x] = struct{}{}
		out = append(out, x)
	}
	return out
}

func contains(xs []string, t string) bool {
	for _, x := range xs {
		if x == t {
			return true
		}
	}
	return false
}

func isAlpha(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}
