package matcher

import (
	"fmt"
	"regexp"

	"github.com/bioshock/gospacy/v3/internal/patternspec"
)

// FromPatternDict registers a pattern in the Python-dict shape spaCy
// uses on disk and in `nlp.add_pipe("entity_ruler").add_patterns([...])`
// calls. Each pattern is a []map[string]any (one map per token
// position); each map's keys are spaCy attribute names (ORTH / LOWER /
// TAG / POS / DEP / LEMMA / ENT_TYPE / IS_SPACE / IS_STOP / IS_ALPHA /
// IS_PUNCT / IS_DIGIT / LIKE_NUM) and values follow the standard shapes:
//
//	"value"                  → scalar equality
//	{"IN":      ["a", "b"]}  → set membership
//	{"NOT_IN":  ["a", "b"]}  → negated set (TAG / DEP only)
//	{"REGEX":   "..."}       → regex (LOWER only)
//	true / false             → IS_* boolean flag
//
// Multiple alternatives for one key are passed as multiple patterns.
// Returns an error if any key is unsupported (Tier 1 ships equality
// only — quantifier OPs and FUZZY are NOT_YET_PORTED) or any value
// has the wrong shape.
//
// Example:
//
//	m.FromPatternDict("AI", [][]map[string]any{
//	    {
//	        {"LOWER": map[string]any{"IN": []any{"artificial", "ai"}}},
//	        {"LOWER": "intelligence"},
//	    },
//	})
func (m *Matcher) FromPatternDict(key string, patterns [][]map[string]any) error {
	specsForAdd := make([][]TokenSpec, 0, len(patterns))
	for pi, pat := range patterns {
		specs := make([]TokenSpec, len(pat))
		for ti, tokDict := range pat {
			s, err := parseTokenDict(tokDict)
			if err != nil {
				return fmt.Errorf("FromPatternDict(%q): pattern %d, token %d: %w", key, pi, ti, err)
			}
			specs[ti] = s
		}
		specsForAdd = append(specsForAdd, specs)
	}
	return m.Add(key, specsForAdd...)
}

// parseTokenDict converts one Python-shape token dict into a TokenSpec.
// Unknown keys, unsupported value shapes, or quantifier OPs (Tier 2)
// return an error rather than silently skipping — Matcher's API
// promise is "I match exactly what you wrote".
func parseTokenDict(d map[string]any) (TokenSpec, error) {
	var s TokenSpec
	for k, v := range d {
		switch k {
		case "ORTH":
			if str, ok := v.(string); ok {
				s.Orth = str
			} else if in, ok := patternspec.ExtractInList(v); ok {
				s.OrthIn = in
			} else {
				return s, fmt.Errorf("ORTH: unsupported value %T (want string or {IN: [...]})", v)
			}
		case "LOWER":
			if str, ok := v.(string); ok {
				s.Lower = str
			} else if in, ok := patternspec.ExtractInList(v); ok {
				s.LowerIn = in
			} else if rx, ok := patternspec.ExtractRegexString(v); ok {
				compiled, err := regexp.Compile(rx)
				if err != nil {
					return s, fmt.Errorf("LOWER: invalid REGEX %q: %w", rx, err)
				}
				s.LowerRegex = compiled
			} else {
				return s, fmt.Errorf("LOWER: unsupported value %T (want string, {IN: [...]} or {REGEX: \"...\"})", v)
			}
		case "TAG":
			if str, ok := v.(string); ok {
				s.Tag = str
			} else if in, ok := patternspec.ExtractInList(v); ok {
				s.TagIn = in
			} else if notIn, ok := patternspec.ExtractNotInList(v); ok {
				s.TagNotIn = notIn
			} else {
				return s, fmt.Errorf("TAG: unsupported value %T", v)
			}
		case "POS":
			if str, ok := v.(string); ok {
				s.Pos = str
			} else if in, ok := patternspec.ExtractInList(v); ok {
				s.PosIn = in
			} else {
				return s, fmt.Errorf("POS: unsupported value %T", v)
			}
		case "DEP":
			if str, ok := v.(string); ok {
				s.Dep = str
			} else if in, ok := patternspec.ExtractInList(v); ok {
				s.DepIn = in
			} else if notIn, ok := patternspec.ExtractNotInList(v); ok {
				s.DepNotIn = notIn
			} else {
				return s, fmt.Errorf("DEP: unsupported value %T", v)
			}
		case "LEMMA":
			if str, ok := v.(string); ok {
				s.Lemma = str
			} else if in, ok := patternspec.ExtractInList(v); ok {
				s.LemmaIn = in
			} else {
				return s, fmt.Errorf("LEMMA: unsupported value %T", v)
			}
		case "ENT_TYPE":
			if str, ok := v.(string); ok {
				s.EntType = str
			} else if in, ok := patternspec.ExtractInList(v); ok {
				s.EntTypeIn = in
			} else {
				return s, fmt.Errorf("ENT_TYPE: unsupported value %T", v)
			}
		case "IS_SPACE":
			b, ok := v.(bool)
			if !ok {
				return s, fmt.Errorf("IS_SPACE: want bool, got %T", v)
			}
			s.IsSpace = boolToTriState(b)
		case "IS_STOP":
			b, ok := v.(bool)
			if !ok {
				return s, fmt.Errorf("IS_STOP: want bool, got %T", v)
			}
			s.IsStop = boolToTriState(b)
		case "IS_ALPHA":
			b, ok := v.(bool)
			if !ok {
				return s, fmt.Errorf("IS_ALPHA: want bool, got %T", v)
			}
			s.IsAlpha = boolToTriState(b)
		case "IS_PUNCT":
			b, ok := v.(bool)
			if !ok {
				return s, fmt.Errorf("IS_PUNCT: want bool, got %T", v)
			}
			s.IsPunct = boolToTriState(b)
		case "IS_DIGIT":
			b, ok := v.(bool)
			if !ok {
				return s, fmt.Errorf("IS_DIGIT: want bool, got %T", v)
			}
			s.IsDigit = boolToTriState(b)
		case "LIKE_NUM":
			b, ok := v.(bool)
			if !ok {
				return s, fmt.Errorf("LIKE_NUM: want bool, got %T", v)
			}
			s.LikeNum = boolToTriState(b)
		case "OP":
			// Quantifier operators are Tier 2 (Thompson NFA build).
			// Fail loud rather than silently degrade to literal match.
			return s, fmt.Errorf("OP: quantifier operators (?, *, +, !, {n,m}) are NOT_YET_PORTED — see Matcher Tier 2")
		default:
			return s, fmt.Errorf("unsupported pattern key %q (supported: ORTH, LOWER, TAG, POS, DEP, LEMMA, ENT_TYPE, IS_SPACE, IS_STOP, IS_ALPHA, IS_PUNCT, IS_DIGIT, LIKE_NUM)", k)
		}
	}
	return s, nil
}

func boolToTriState(b bool) int8 {
	if b {
		return 1
	}
	return -1
}
