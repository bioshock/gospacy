// Package doc holds the runtime container produced by tokenizer + pipeline
// components. Doc is the top-level structure; Token is one populated entry;
// Span is a half-open [start,end) slice over Doc.Tokens.
//
// Field naming follows the Python `spacy.tokens.Token` C struct: hash-valued
// attributes (orth, lemma, ...) carry the suffix-free name; their stringified
// counterparts are recoverable via the parent Vocab.
package doc

// Token is one entry in a Doc. Mirrors `spacy.tokens.token.Token` C-layout
// fields. Construction: zero-valued Tokens are filled in by the tokenizer and
// then by each pipeline component in order (tok2vec → tagger →
// attribute_ruler → lemmatizer → parser → ner). Hash-valued fields refer to
// the parent Vocab's StringStore.
type Token struct {
	// Lexical hashes (resolve via Vocab.StringStore.Lookup).
	Orth   uint64
	Lemma  uint64
	Norm   uint64
	Lower  uint64
	Prefix uint64
	Suffix uint64

	// Inline strings kept for round-trip text reconstruction.
	Text       string // surface form, exactly the substring from the source text
	Whitespace string // trailing whitespace consumed from the source after Text
	Shape      string // e.g. "Xxxx" for "Hello"; cached because shape() is fast but
	//                  used by FeatureExtractor every forward pass

	// Tagger output.
	Tag uint64 // fine-grained tag hash (e.g. "NN", "VBZ")
	POS uint64 // coarse-grained Universal POS hash (e.g. "NOUN", "VERB")

	// AttributeRuler output (morphology as "Feat=Val|Feat=Val" form).
	Morph string

	// Parser output. Dep is the StringStore hash of the dep label (e.g.
	// "nsubj"); Head is the index of the head token in Doc.Tokens (self if
	// root). SentStart: 1 = sentence start, 0 = not a start, -1 = unknown.
	Dep       uint64
	Head      int  // index into Doc.Tokens; self if root
	SentStart int8 // -1 unknown, 0 not a sentence start, 1 sentence start

	// NER output. Not populated in v0.1 — `ner` is listed in
	// NOT_YET_PORTED.md. EntIOB: 0 missing, 1 I-, 2 O, 3 B-.
	EntIOB  uint8 // 0 missing, 1 I-, 2 O, 3 B-
	EntType uint64

	// Offsets.
	Idx int // Unicode rune offset of Text in Doc.Text
}

// TextWithWS returns Text concatenated with its trailing Whitespace. Mirrors
// spaCy's `Token.text_with_ws`. Used to reconstruct Doc.Text() in O(n).
func (t Token) TextWithWS() string { return t.Text + t.Whitespace }
