// Package tokenizer implements gospacy's rule-based tokenizer. The engine
// itself is language-independent; per-language data (prefix/suffix/infix
// patterns + the exception table) is supplied via Rules, typically built by
// lang/<code>/MakeRules. Algorithm mirrors spacy.Tokenizer (tokenizer.pyx)
// with the same special-case-first / split-affixes / collect-token order.
package tokenizer

import (
	"fmt"

	"github.com/dlclark/regexp2"
)

// SpecialPiece is one orth-string emitted by a special-case rule.
// Norm carries the optional NORM-attribute override Python attaches to clitic
// sub-tokens (e.g. {Orth:"n't", Norm:"not"}); empty string means "no override,
// fall back to the lexeme's Lower hash". See ToDoc for application.
type SpecialPiece struct {
	Orth  string
	Norm  string
	Lemma string
	Tag   string
	POS   string
}

// Span is a half-open [Start, End) UTF-8 byte range inside a string.
type Span struct{ Start, End int }

// RulesInput is the raw, uncompiled tokenizer data; compiled once by NewRules.
type RulesInput struct {
	Prefixes   []string
	Suffixes   []string
	Infixes    []string
	TokenMatch string
	URLMatch   string
	Specials   map[string][]SpecialPiece
}

// Rules holds the compiled patterns. Build once, reuse.
type Rules struct {
	prefixes []*regexp2.Regexp
	suffixes []*regexp2.Regexp
	infixes  []*regexp2.Regexp
	tokenRe  *regexp2.Regexp
	urlRe    *regexp2.Regexp
	specials map[string][]SpecialPiece
}

// NewRules compiles every pattern in in and returns the immutable Rules.
func NewRules(in RulesInput) (*Rules, error) {
	r := &Rules{specials: in.Specials}
	for i, p := range in.Prefixes {
		re, err := regexp2.Compile(p, 0)
		if err != nil {
			return nil, fmt.Errorf("NewRules: prefix[%d] %q: %w", i, p, err)
		}
		r.prefixes = append(r.prefixes, re)
	}
	for i, p := range in.Suffixes {
		// Anchor each suffix pattern to the end of the string, mirroring
		// spaCy's suffix_search which joins all patterns as "p1$|p2$|…"
		// and uses re.search(). Without the anchor, regexp2's FindStringMatch
		// returns the first (leftmost) match, which may not be at the string
		// end; with "$" the engine is forced to find a match ending exactly
		// at the string terminus, enabling correct alternation resolution
		// (e.g. "mm$" wins over "m$" in "35mm").
		re, err := regexp2.Compile(p+"$", 0)
		if err != nil {
			return nil, fmt.Errorf("NewRules: suffix[%d] %q: %w", i, p, err)
		}
		r.suffixes = append(r.suffixes, re)
	}
	for i, p := range in.Infixes {
		re, err := regexp2.Compile(p, 0)
		if err != nil {
			return nil, fmt.Errorf("NewRules: infix[%d] %q: %w", i, p, err)
		}
		r.infixes = append(r.infixes, re)
	}
	if in.TokenMatch != "" {
		re, err := regexp2.Compile(in.TokenMatch, 0)
		if err != nil {
			return nil, fmt.Errorf("NewRules: token_match: %w", err)
		}
		r.tokenRe = re
	}
	if in.URLMatch != "" {
		re, err := regexp2.Compile(in.URLMatch, 0)
		if err != nil {
			return nil, fmt.Errorf("NewRules: url_match: %w", err)
		}
		r.urlRe = re
	}
	return r, nil
}

// FindPrefix returns the longest prefix matched by any prefix pattern,
// or "" + false if nothing matches.
// regexp2 returns rune (character) offsets, so we slice using []rune.
func (r *Rules) FindPrefix(s string) (string, bool) {
	bestEnd := 0 // rune count
	for _, re := range r.prefixes {
		m, _ := re.FindStringMatch(s)
		if m != nil && m.Index == 0 && m.Length > bestEnd {
			bestEnd = m.Length
		}
	}
	if bestEnd == 0 {
		return "", false
	}
	return string([]rune(s)[:bestEnd]), true
}

// FindSuffix returns the longest suffix matched, or "" + false.
// Each suffix pattern is compiled with a "$" anchor (see NewRules), so every
// match already ends at the string terminus. We pick the match that starts
// earliest (leftmost), which gives the longest suffix — mirroring spaCy's
// suffix_search behaviour.
// regexp2 returns rune (character) offsets, so we slice using []rune.
func (r *Rules) FindSuffix(s string) (string, bool) {
	runes := []rune(s)
	runeLen := len(runes)
	bestStart := runeLen // rune index
	for _, re := range r.suffixes {
		m, _ := re.FindStringMatch(s)
		// With "$" anchored patterns, any match returned ends at runeLen.
		// We iterate all non-overlapping matches to find the one starting
		// earliest (in case a pattern has multiple "$"-terminated matches,
		// though in practice each pattern produces at most one).
		for m != nil {
			if m.Index+m.Length == runeLen && m.Index < bestStart {
				bestStart = m.Index
			}
			m, _ = re.FindNextMatch(m)
		}
	}
	if bestStart == runeLen {
		return "", false
	}
	return string(runes[bestStart:]), true
}

// FindInfixes returns every infix-pattern match span in s, in left-to-right
// order. Spans from different patterns may overlap.
func (r *Rules) FindInfixes(s string) []Span {
	var out []Span
	for _, re := range r.infixes {
		m, _ := re.FindStringMatch(s)
		for m != nil {
			out = append(out, Span{Start: m.Index, End: m.Index + m.Length})
			m, _ = re.FindNextMatch(m)
		}
	}
	return out
}

// IsTokenMatch returns true if the entire string matches token_match.
// regexp2 returns rune offsets, so we compare against rune count.
func (r *Rules) IsTokenMatch(s string) bool {
	if r.tokenRe == nil {
		return false
	}
	m, _ := r.tokenRe.FindStringMatch(s)
	return m != nil && m.Index == 0 && m.Length == len([]rune(s))
}

// IsURLMatch returns true if s matches url_match.
// regexp2 returns rune offsets, so we compare against rune count.
func (r *Rules) IsURLMatch(s string) bool {
	if r.urlRe == nil {
		return false
	}
	m, _ := r.urlRe.FindStringMatch(s)
	return m != nil && m.Index == 0 && m.Length == len([]rune(s))
}

// Special looks up s in the specials map.
func (r *Rules) Special(s string) ([]SpecialPiece, bool) {
	p, ok := r.specials[s]
	return p, ok
}
