package doc

import "github.com/bioshock/gospacy/v3/vocab"

// IsStop reports whether t is an English stop word per spaCy's
// lang/en/stop_words.STOP_WORDS. The check uses t.Lower (already
// lower-cased and interned by the tokenizer); v.StringStore() resolves
// the hash back to a string and the result is looked up in vocab.IsStopEN.
//
// Returns false if t.Lower cannot be resolved (Lower not yet populated,
// or the store has not interned the lower-cased form).
//
// English-only; spaCy stores per-language stop lists on the Vocab, but
// gospacy v0.1 ships only English so we short-circuit to the English set.
func (t Token) IsStop(v *vocab.Vocab) bool {
	if v == nil {
		return false
	}
	lower, ok := v.StringStore().Lookup(t.Lower)
	if !ok {
		return false
	}
	return vocab.IsStopEN(lower)
}
