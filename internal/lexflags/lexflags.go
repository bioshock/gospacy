// Package lexflags implements the boolean lexical-attribute predicates
// spaCy's Token exposes: IS_ALPHA, IS_PUNCT, IS_DIGIT, LIKE_NUM, etc.
// These are intentionally string→bool (no Token coupling) so they can
// be called from the matcher engine, the comprehensive-analyzer
// example, and future callers without dragging the *doc.Token type in.
//
// Ports spacy/lang/en/lex_attrs.py and the corresponding global helpers
// in spacy/lang/lex_attrs.py.
package lexflags

import "unicode"

// IsAlpha reports whether every rune in s is a unicode letter, and s
// is non-empty. Mirrors Token.is_alpha.
func IsAlpha(s string) bool {
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

// IsPunct reports whether every rune in s is a unicode punctuation
// or symbol character, and s is non-empty. Mirrors Token.is_punct.
// spaCy treats both Po/Pc/Pd/Ps/Pe/Pi/Pf categories as punct, so we
// use unicode.IsPunct which covers all six.
func IsPunct(s string) bool {
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

// IsDigit reports whether every rune in s is a unicode digit, and s
// is non-empty. Mirrors Token.is_digit.
func IsDigit(s string) bool {
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

// numWords is the cardinal-number word set spaCy's English
// lex_attrs.like_num matches. Source:
// spacy/lang/en/lex_attrs.py:_num_words.
var numWords = map[string]struct{}{
	"zero": {}, "one": {}, "two": {}, "three": {}, "four": {},
	"five": {}, "six": {}, "seven": {}, "eight": {}, "nine": {},
	"ten": {}, "eleven": {}, "twelve": {}, "thirteen": {},
	"fourteen": {}, "fifteen": {}, "sixteen": {}, "seventeen": {},
	"eighteen": {}, "nineteen": {}, "twenty": {}, "thirty": {},
	"forty": {}, "fifty": {}, "sixty": {}, "seventy": {},
	"eighty": {}, "ninety": {}, "hundred": {}, "thousand": {},
	"million": {}, "billion": {}, "trillion": {}, "quadrillion": {},
	"quintillion": {}, "sextillion": {}, "septillion": {},
	"octillion": {}, "nonillion": {}, "decillion": {},
	"gajillion": {}, "bazillion": {},
}

// ordinalWords is the ordinal-number word set
// (spacy/lang/en/lex_attrs.py:_ordinal_words).
var ordinalWords = map[string]struct{}{
	"first": {}, "second": {}, "third": {}, "fourth": {},
	"fifth": {}, "sixth": {}, "seventh": {}, "eighth": {},
	"ninth": {}, "tenth": {}, "eleventh": {}, "twelfth": {},
	"thirteenth": {}, "fourteenth": {}, "fifteenth": {},
	"sixteenth": {}, "seventeenth": {}, "eighteenth": {},
	"nineteenth": {}, "twentieth": {}, "thirtieth": {},
	"fortieth": {}, "fiftieth": {}, "sixtieth": {},
	"seventieth": {}, "eightieth": {}, "ninetieth": {},
	"hundredth": {}, "thousandth": {}, "millionth": {},
	"billionth": {}, "trillionth": {}, "quadrillionth": {},
	"quintillionth": {}, "sextillionth": {}, "septillionth": {},
	"octillionth": {}, "nonillionth": {}, "decillionth": {},
	"gajillionth": {}, "bazillionth": {},
}

// LikeNum mirrors spacy/lang/en/lex_attrs.py:like_num exactly. True
// for digit strings (with optional leading sign and stripped commas /
// periods), simple fractions ("3/4"), cardinal number words, ordinal
// number words, and "<digits>st|nd|rd|th" forms (e.g. "21st").
func LikeNum(text string) bool {
	if text == "" {
		return false
	}
	// Strip optional leading sign (+, -, ±, ~). spaCy uses startswith
	// on a tuple; we mirror that.
	if r := []rune(text)[0]; r == '+' || r == '-' || r == '±' || r == '~' {
		text = string([]rune(text)[1:])
	}
	stripped := stripCommasAndPeriods(text)
	if isAllDigits(stripped) {
		return true
	}
	// Simple fraction: exactly one '/', both sides digit.
	if countByte(text, '/') == 1 {
		for i := 0; i < len(text); i++ {
			if text[i] == '/' {
				num := text[:i]
				denom := text[i+1:]
				if isAllDigits(num) && isAllDigits(denom) {
					return true
				}
				break
			}
		}
	}
	lower := toLowerASCII(text)
	if _, ok := numWords[lower]; ok {
		return true
	}
	if _, ok := ordinalWords[lower]; ok {
		return true
	}
	// "<digits>st|nd|rd|th" suffix form (e.g. "21st", "32nd").
	if len(lower) >= 3 {
		suffix := lower[len(lower)-2:]
		if suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th" {
			if isAllDigits(lower[:len(lower)-2]) {
				return true
			}
		}
	}
	return false
}

// isAllDigits — true if s is non-empty and every byte is ASCII 0-9.
// Matches Python's str.isdigit semantics on the spaCy code path
// (text is ASCII after the strip pass).
func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

func stripCommasAndPeriods(s string) string {
	keep := s
	out := make([]byte, 0, len(keep))
	for i := 0; i < len(keep); i++ {
		if keep[i] == ',' || keep[i] == '.' {
			continue
		}
		out = append(out, keep[i])
	}
	return string(out)
}

func countByte(s string, b byte) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			n++
		}
	}
	return n
}

// toLowerASCII lowercases ASCII letters in s. Non-ASCII bytes pass
// through unchanged. Sufficient because the comparison sets above are
// ASCII-only.
func toLowerASCII(s string) string {
	out := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		out[i] = c
	}
	return string(out)
}
