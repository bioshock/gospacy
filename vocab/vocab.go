package vocab

// Vocab owns the StringStore and caches one Lexeme per orth hash. Mirrors
// spacy.Vocab — the container shared across every Doc/Token in a pipeline.
//
// Not safe for concurrent use. Get is a lazy-write under the hood: it
// inserts a new Lexeme into the unsynchronized lexemes map on cache miss
// and (via NewLexeme) interns five strings into the StringStore. Callers
// that need parallelism must hold one *Vocab per goroutine; for the
// pipeline-level constraint see bundle.Bundle.Pipe's godoc.
type Vocab struct {
	strings *StringStore
	lexemes map[uint64]*Lexeme
	vectors *Vectors
}

// NewVocab returns an empty Vocab.
func NewVocab() *Vocab {
	return &Vocab{
		strings: NewStringStore(),
		lexemes: make(map[uint64]*Lexeme),
	}
}

// Clone returns a deep copy of the Vocab for a parallel-worker handoff. The
// returned Vocab has its own StringStore (deep-copied) and its own lexemes
// map; the source and clone can independently lazy-intern new strings without
// racing. The Vectors pointer is shared by reference because *Vectors is
// immutable post-load (LoadVectorsDir is the only mutator and runs at
// FromDisk time).
//
// Used by bundle.Bundle.Clone for the goroutine-per-Bundle parallelism
// pattern. Safe to call once before launching workers; NOT safe to call
// concurrently with Get on the source Vocab.
func (v *Vocab) Clone() *Vocab {
	dst := &Vocab{
		strings: v.strings.Clone(),
		lexemes: make(map[uint64]*Lexeme, len(v.lexemes)),
		vectors: v.vectors,
	}
	for k, lex := range v.lexemes {
		dst.lexemes[k] = lex
	}
	return dst
}

// StringStore returns the underlying StringStore.
func (v *Vocab) StringStore() *StringStore { return v.strings }

// Vectors returns the bundle-loaded *Vectors, or nil if vectors haven't been
// loaded yet (e.g. en_core_web_sm-shape bundles where vectors are empty).
func (v *Vocab) Vectors() *Vectors { return v.vectors }

// SetVectors attaches a *Vectors to the Vocab. Called by bundle.FromDisk
// after LoadVectorsDir. The Vocab does not own the *Vectors lifecycle; the
// bundle keeps a reference too.
func (v *Vocab) SetVectors(vec *Vectors) { v.vectors = vec }

// Get returns the Lexeme for str, computing and caching it if necessary.
// Always returns a non-nil pointer.
//
// Despite the name, Get writes: on cache miss it inserts a fresh Lexeme
// into v.lexemes AND interns five strings (orth, prefix, suffix, shape,
// lower) into v.strings via NewLexeme. Both maps are unsynchronized; not
// safe to call concurrently on a shared *Vocab.
func (v *Vocab) Get(str string) *Lexeme {
	h := v.strings.Hash(str)
	if lex, ok := v.lexemes[h]; ok {
		return lex
	}
	lex := NewLexeme(v.strings, str)
	v.lexemes[h] = lex
	return lex
}

// GetByHash returns the cached Lexeme for an orth hash. Returns nil if the
// hash has not been seen via Get. Does NOT auto-compute (matches spaCy's
// `vocab[hash]` raising KeyError for unknown hashes after store-only adds).
func (v *Vocab) GetByHash(h uint64) *Lexeme {
	return v.lexemes[h]
}
