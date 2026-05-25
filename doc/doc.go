package doc

import (
	"strings"

	"github.com/bioshock/gospacy/v3/vocab"
)

// Doc is the runtime container produced by the tokenizer and successively
// annotated by each pipeline component. Mirrors `spacy.tokens.doc.Doc`. The
// raw input text is preserved verbatim in source so that Text() round-trips.
//
// Tokens is exposed (not via a getter) because every pipeline component
// mutates it in place. This matches spaCy's `for token in doc: token.tag = ...`
// pattern and avoids a get/set boundary in the hot path.
type Doc struct {
	Vocab  *vocab.Vocab
	source string
	Tokens []Token

	// CSR children index — built lazily from Tokens[].Head on the first
	// ChildrenOf / SubtreeOf call. childStart has length NumTokens+1;
	// childStart[i]:childStart[i+1] are the child indices of token i,
	// stored in childIdx in ascending token order. nil until built.
	//
	// Cache invariant: built once per Doc after Head writes complete (in
	// practice: after the parser pass). Any post-build mutation of
	// Token.Head leaves the cache stale. Since gospacy is
	// inference-only and Head is set once by the parser, this is safe.
	childStart []int32
	childIdx   []int32
}

// NewDoc returns an empty Doc bound to v with source text preserved for
// Text() reconstruction. Tokens is nil; the tokenizer (or a downstream
// component) fills it in.
func NewDoc(v *vocab.Vocab, text string) *Doc {
	return &Doc{Vocab: v, source: text}
}

// NumTokens returns len(d.Tokens). Provided as a method for parity with
// Python's `len(doc)` ergonomics.
func (d *Doc) NumTokens() int { return len(d.Tokens) }

// Text reconstructs the source text by concatenating each token's TextWithWS.
// For a well-formed Doc this equals the original input passed to NewDoc.
func (d *Doc) Text() string {
	if len(d.Tokens) == 0 {
		return d.source
	}
	var b strings.Builder
	b.Grow(len(d.source))
	for i := range d.Tokens {
		b.WriteString(d.Tokens[i].TextWithWS())
	}
	return b.String()
}

// Source returns the raw text passed to NewDoc (before tokenization). Useful
// when the Tokens slice is empty or partially filled.
func (d *Doc) Source() string { return d.source }
