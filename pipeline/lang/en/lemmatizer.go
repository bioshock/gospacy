// Package en wires English-specific behavior into the Lemmatizer. Mirrors
// spacy/lang/en/lemmatizer.py — currently only is_base_form.
package en

import (
	"strings"

	"github.com/bioshock/gospacy/v3/doc"
)

// IsBaseForm returns true when the token's morphology indicates it is already
// in its base form, so the Lemmatizer can skip suffix-rule processing.
// Port of EnglishLemmatizer.is_base_form (spacy/lang/en/lemmatizer.py).
func IsBaseForm(tok *doc.Token, univPos string) bool {
	morph := parseMorph(tok.Morph)
	number := morph["Number"]
	verbForm := morph["VerbForm"]
	tense := morph["Tense"]
	degree := morph["Degree"]
	switch univPos {
	case "noun":
		if number == "Sing" {
			return true
		}
	case "verb":
		if verbForm == "Inf" {
			return true
		}
		if verbForm == "Fin" && tense == "Pres" && number == "" {
			return true
		}
	case "adj":
		if degree == "Pos" {
			return true
		}
	}
	if verbForm == "Inf" {
		return true
	}
	if verbForm == "None" {
		return true
	}
	if degree == "Pos" {
		return true
	}
	return false
}

func parseMorph(s string) map[string]string {
	if s == "" || s == "_" {
		return nil
	}
	out := map[string]string{}
	for _, part := range strings.Split(s, "|") {
		i := strings.IndexByte(part, '=')
		if i < 0 {
			continue
		}
		out[part[:i]] = part[i+1:]
	}
	return out
}
