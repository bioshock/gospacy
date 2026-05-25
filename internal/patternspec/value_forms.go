// Package patternspec implements the shared value-form parsers used
// by the AttributeRuler loader and the public matcher package.
//
// spaCy's Matcher token-pattern values come in a few shapes that
// callers want to recognise once and apply consistently:
//
//	"value"                  — scalar equality
//	{"IN":      ["a", "b"]}  — set membership
//	{"NOT_IN":  ["a", "b"]}  — negated set
//	{"REGEX":   "..."}       — regex (LOWER only)
//	true / false             — IS_* boolean flag
//
// These helpers extract each shape from a generic `any` (the typed
// value coming from msgpack/JSON decode) so the per-key switch in
// each consumer stays small and consistent.
package patternspec

// ExtractInList unwraps the set-membership form {"IN": [s1, s2, ...]}
// into a flat []string. Returns (nil, false) for anything else.
func ExtractInList(v any) ([]string, bool) {
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

// ExtractNotInList unwraps the negated set-membership form
// {"NOT_IN": [s1, s2, ...]}. Same return shape as ExtractInList.
func ExtractNotInList(v any) ([]string, bool) {
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

// ExtractRegexString unwraps the regex form {"REGEX": "pattern"}.
// Returns ("", false) for anything else.
func ExtractRegexString(v any) (string, bool) {
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
