package pipeline

import (
	"fmt"
	"os"
	"regexp"

	"github.com/vmihailenco/msgpack/v5"
)

// arTokenSpec is one token in a pattern's pattern-list. Mirrors the
// {ORTH/LOWER/TAG: value} dict spaCy's Matcher consumes. Supports TAG, ORTH,
// LOWER, and DEP as either a single string (equality), a {"IN": [...]} set
// (membership), or a {"NOT_IN": [...]} negated set; IS_SPACE as a boolean
// flag matcher (true → token's Text is all unicode whitespace).
//
// For each string field exactly one of <Field> (scalar) or <Field>In (set)
// is set, or both are empty when the field is unconstrained.
type arTokenSpec struct {
	Orth     string // empty when not constrained
	OrthIn   []string
	Lower    string // empty when not constrained
	LowerIn  []string
	Tag      string // empty when not constrained
	TagIn    []string
	TagNotIn []string // negated set; non-empty disqualifies tokens whose Tag matches
	Dep      string   // empty when not constrained (matched against Token.Dep)
	DepIn    []string
	DepNotIn []string // negated set; "" sentinel requires Dep != 0
	// IsSpace tri-state: 0 unconstrained, 1 token must be all-whitespace,
	// -1 token must NOT be all-whitespace. Mirrors spaCy's boolean
	// IS_SPACE Matcher flag. en_core_web_md/lg's AR patterns use both
	// polarities (patterns 176/177/178 in md).
	IsSpace int8
	// LowerRegex is nil unless the pattern uses LOWER: {REGEX: "..."}.
	// Tested against strings.ToLower(tok.Text) in matchPattern. md/lg
	// patterns 147/153/163 use this for English contractions ("n't",
	// "nothin'", "y'").
	LowerRegex *regexp.Regexp
}

// extractRegexString unwraps {"REGEX": "<pattern>"}. Returns ("", false)
// for anything else.
func extractRegexString(v any) (string, bool) {
	d, ok := v.(map[string]any)
	if !ok {
		return "", false
	}
	rxAny, ok := d["REGEX"]
	if !ok || len(d) != 1 {
		return "", false
	}
	s, ok := rxAny.(string)
	if !ok {
		return "", false
	}
	return s, true
}

// arAttrs is the {POS, MORPH, LEMMA, TAG} dict written to the matched token.
type arAttrs struct {
	POS   string
	Tag   string
	Lemma string
	Morph string
}

// arPattern is one entry in attribute_ruler/patterns. TokenSpecs has length >= 1
// (the pattern's token-sequence). Index selects which token in a matched span
// gets the attrs (almost always 0).
type arPattern struct {
	TokenSpecs []arTokenSpec
	Attrs      arAttrs
	Index      int
	// True when the pattern uses a key gospacy's minimal matcher does not
	// implement; Apply emits a debug warning once per such pattern and
	// continues. Surfaces unsupported coverage instead of silently skipping.
	Unsupported bool
}

// extractInList unwraps the spaCy Matcher set-membership form
// {"IN": ["a", "b", ...]} into a flat []string. Returns (nil, false) for
// anything else (including {"NOT_IN": ...}, {"REGEX": ...}, plain dicts,
// numeric lists). Used by the per-key switch in loadAttributeRulerPatterns
// to decide whether a pattern is gospacy-supported.
func extractInList(v any) ([]string, bool) {
	d, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	inAny, ok := d["IN"]
	if !ok || len(d) != 1 {
		return nil, false
	}
	list, ok := inAny.([]any)
	if !ok {
		return nil, false
	}
	out := make([]string, 0, len(list))
	for _, x := range list {
		s, ok := x.(string)
		if !ok {
			return nil, false
		}
		out = append(out, s)
	}
	return out, true
}

// extractNotInList mirrors extractInList for the {"NOT_IN": [...]} form.
// Used for TAG with negated set-membership (pattern 170 in en_core_web_sm
// uses TAG: {NOT_IN: ["TO", ""]}).
func extractNotInList(v any) ([]string, bool) {
	d, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	inAny, ok := d["NOT_IN"]
	if !ok || len(d) != 1 {
		return nil, false
	}
	list, ok := inAny.([]any)
	if !ok {
		return nil, false
	}
	out := make([]string, 0, len(list))
	for _, x := range list {
		s, ok := x.(string)
		if !ok {
			return nil, false
		}
		out = append(out, s)
	}
	return out, true
}

// loadAttributeRulerPatterns reads the msgpack list. Each entry is
//
//	{"patterns": [[{TAG/ORTH/...: str}, ...]], "attrs": {POS/MORPH/...: str}, "index": int}.
//
// We flatten patterns[0] (spaCy stores them as list-of-list for multi-rule
// support; gospacy treats each entry as one rule).
func loadAttributeRulerPatterns(path string) ([]arPattern, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("attributeruler: read patterns: %w", err)
	}
	var raw []map[string]any
	if err := msgpack.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("attributeruler: decode patterns: %w", err)
	}
	out := make([]arPattern, 0, len(raw))
	for i, entry := range raw {
		patAny, ok := entry["patterns"]
		if !ok {
			return nil, fmt.Errorf("attributeruler: pattern %d missing 'patterns'", i)
		}
		patList, ok := patAny.([]any)
		if !ok || len(patList) == 0 {
			return nil, fmt.Errorf("attributeruler: pattern %d 'patterns' not a non-empty list (%T)", i, patAny)
		}
		// patList[0] is the actual rule (list of token-specs). spaCy supports
		// alternative patterns via the outer list; for the en_core_web_sm
		// patterns file every entry has exactly one rule.
		rule, ok := patList[0].([]any)
		if !ok {
			return nil, fmt.Errorf("attributeruler: pattern %d rule not a list (%T)", i, patList[0])
		}
		specs := make([]arTokenSpec, len(rule))
		unsupported := false
		for j, tokAny := range rule {
			tokDict, ok := tokAny.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("attributeruler: pattern %d token %d not a dict (%T)", i, j, tokAny)
			}
			for k, v := range tokDict {
				switch k {
				case "ORTH":
					if s, isStr := v.(string); isStr {
						specs[j].Orth = s
					} else if in, isIn := extractInList(v); isIn {
						specs[j].OrthIn = in
					} else {
						unsupported = true
					}
				case "LOWER":
					if s, isStr := v.(string); isStr {
						specs[j].Lower = s
					} else if in, isIn := extractInList(v); isIn {
						specs[j].LowerIn = in
					} else if rx, isRx := extractRegexString(v); isRx {
						compiled, err := regexp.Compile(rx)
						if err != nil {
							// Malformed regex on disk: surface as unsupported
							// rather than aborting the whole bundle load.
							unsupported = true
						} else {
							specs[j].LowerRegex = compiled
						}
					} else {
						unsupported = true
					}
				case "TAG":
					if s, isStr := v.(string); isStr {
						specs[j].Tag = s
					} else if in, isIn := extractInList(v); isIn {
						specs[j].TagIn = in
					} else if notIn, isNotIn := extractNotInList(v); isNotIn {
						specs[j].TagNotIn = notIn
					} else {
						unsupported = true
					}
				case "DEP":
					if s, isStr := v.(string); isStr {
						specs[j].Dep = s
					} else if in, isIn := extractInList(v); isIn {
						specs[j].DepIn = in
					} else if notIn, isNotIn := extractNotInList(v); isNotIn {
						specs[j].DepNotIn = notIn
					} else {
						unsupported = true
					}
				case "IS_SPACE":
					if b, isBool := v.(bool); isBool {
						if b {
							specs[j].IsSpace = 1
						} else {
							specs[j].IsSpace = -1
						}
					} else {
						unsupported = true
					}
				default:
					unsupported = true
				}
			}
		}
		attrs := arAttrs{}
		if a, ok := entry["attrs"].(map[string]any); ok {
			attrs.POS, _ = a["POS"].(string)
			attrs.Tag, _ = a["TAG"].(string)
			attrs.Lemma, _ = a["LEMMA"].(string)
			attrs.Morph, _ = a["MORPH"].(string)
		}
		idx := 0
		switch x := entry["index"].(type) {
		case int64:
			idx = int(x)
		case uint64:
			idx = int(x)
		case int:
			idx = x
		}
		out = append(out, arPattern{
			TokenSpecs:  specs,
			Attrs:       attrs,
			Index:       idx,
			Unsupported: unsupported,
		})
	}
	return out, nil
}
