package workflows

import (
	"strings"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// isPunctToken approximates spaCy's Token.is_punct by checking the cached
// Lexeme behind t.Orth. Returns false if the lexeme was never interned
// (which shouldn't happen for tokens emitted by the tokenizer — every
// surface form goes through Vocab.Get).
func isPunctToken(t doc.Token, v *vocab.Vocab) bool {
	if v == nil {
		return false
	}
	lex := v.GetByHash(t.Orth)
	if lex == nil {
		return false
	}
	return lex.IsPunct
}

// morphHas reports whether t.Morph contains "Feat=Val" as one of its
// pipe-separated entries. The Morph string follows spaCy's serialization
// (Feat=Val|Feat=Val); empty string is "" (no morph).
func morphHas(morph, feat, val string) bool {
	if morph == "" {
		return false
	}
	target := feat + "=" + val
	for _, kv := range strings.Split(morph, "|") {
		if kv == target {
			return true
		}
	}
	return false
}

// vocabForDoc resolves the *vocab.Vocab from a Doc (centralized so workflow
// implementations don't reach into d.Vocab directly more than once).
func vocabForDoc(d *doc.Doc) *vocab.Vocab { return d.Vocab }
