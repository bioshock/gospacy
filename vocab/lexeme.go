package vocab

import (
	"strings"
	"unicode"
)

// Lexeme holds per-string lexical attributes. Equivalent to spaCy's Cython
// Lexeme class. Construct via NewLexeme; Vocab caches one Lexeme per orth hash.
type Lexeme struct {
	// Interned hashes (resolve via Vocab.StringStore).
	Orth   uint64
	Prefix uint64
	Suffix uint64
	Shape  uint64
	Lower  uint64

	// Boolean flags.
	IsAlpha bool
	IsDigit bool
	IsPunct bool
	IsSpace bool
	IsLower bool
	IsUpper bool
	IsTitle bool
	IsASCII bool
	LikeNum bool
}

// Default prefix/suffix lengths used by spaCy's HashEmbed feature extraction.
// spaCy uses prefix length 1 and suffix length 3, matching
// spacy/lang/lex_attrs.py's word_shape and the built-in prefix_search rules.
const (
	defaultPrefixLen = 1
	defaultSuffixLen = 3
)

// NewLexeme computes lexical attributes for str, interns the relevant
// substrings into the StringStore, and returns a populated Lexeme.
func NewLexeme(s *StringStore, str string) *Lexeme {
	lex := &Lexeme{
		Orth:   s.Add(str),
		Prefix: s.Add(prefixOf(str, defaultPrefixLen)),
		Suffix: s.Add(suffixOf(str, defaultSuffixLen)),
		Shape:  s.Add(wordShape(str)),
		Lower:  s.Add(strings.ToLower(str)),
	}
	lex.IsAlpha = isAllAlpha(str)
	lex.IsDigit = isAllDigit(str)
	lex.IsPunct = isAllPunct(str)
	lex.IsSpace = isAllSpace(str)
	lex.IsLower = isAllLowerLetters(str)
	lex.IsUpper = isAllUpperLetters(str)
	lex.IsTitle = isTitleCase(str)
	lex.IsASCII = isAllASCII(str)
	lex.LikeNum = isAllDigit(str) // minimal port; full like_num handles "ten", "1.5", "1,000"
	return lex
}

func prefixOf(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

func suffixOf(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[len(r)-n:])
}

// wordShape mirrors spacy.lang.lex_attrs.word_shape: per-char map
// (upper→X, lower→x, digit→d, other→keep), then for strings of length >=4
// collapse runs of 4+ identical chars down to 4.
func wordShape(s string) string {
	if s == "" {
		return ""
	}
	r := []rune(s)
	mapped := make([]rune, len(r))
	for i, c := range r {
		switch {
		case unicode.IsDigit(c):
			mapped[i] = 'd'
		case unicode.IsUpper(c):
			mapped[i] = 'X'
		case unicode.IsLower(c):
			mapped[i] = 'x'
		default:
			mapped[i] = c
		}
	}
	if len(mapped) < 4 {
		return string(mapped)
	}
	var out []rune
	run := 1
	for i := 0; i < len(mapped); i++ {
		if i > 0 && mapped[i] == mapped[i-1] {
			run++
		} else {
			run = 1
		}
		if run <= 4 {
			out = append(out, mapped[i])
		}
	}
	return string(out)
}

func isAllAlpha(s string) bool {
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

func isAllDigit(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// isAllPunct mirrors spaCy's lex_attrs.is_punct: every rune's Unicode
// category must start with "P" (Pc/Pd/Pe/Pf/Pi/Po/Ps). Symbols (categories
// S*: Sc/Sm/Sk/So) are NOT punctuation in spaCy — "$", "+", "<", "£" all
// return is_punct=False there. Go's unicode.IsPunct matches P* exactly,
// which is what we want.
func isAllPunct(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsPunct(r) {
			return false
		}
	}
	return true
}

func isAllSpace(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return true
}

func isAllLowerLetters(s string) bool {
	if s == "" {
		return false
	}
	hasLetter := false
	for _, r := range s {
		if unicode.IsLetter(r) {
			hasLetter = true
			if !unicode.IsLower(r) {
				return false
			}
		}
	}
	return hasLetter
}

func isAllUpperLetters(s string) bool {
	if s == "" {
		return false
	}
	hasLetter := false
	for _, r := range s {
		if unicode.IsLetter(r) {
			hasLetter = true
			if !unicode.IsUpper(r) {
				return false
			}
		}
	}
	return hasLetter
}

// isTitleCase mirrors Python's str.istitle(): the string is title-case if
// cased characters follow uncased characters (or are at the string start) and
// are uppercase, while all other cased characters are lowercase.
// Equivalent to: each "word" (maximal run of letters) starts with uppercase
// and continues with all lowercase. Strings with no cased characters are
// not title-case.
func isTitleCase(s string) bool {
	if s == "" {
		return false
	}
	hasCased := false
	prevWasLetter := false
	for _, c := range s {
		if unicode.IsLetter(c) {
			hasCased = true
			if !prevWasLetter {
				// Start of a new "word": must be uppercase.
				if !unicode.IsUpper(c) {
					return false
				}
			} else {
				// Continuation of a word: must be lowercase.
				if !unicode.IsLower(c) {
					return false
				}
			}
			prevWasLetter = true
		} else {
			prevWasLetter = false
		}
	}
	return hasCased
}

func isAllASCII(s string) bool {
	for _, r := range s {
		if r > 127 {
			return false
		}
	}
	return true
}
