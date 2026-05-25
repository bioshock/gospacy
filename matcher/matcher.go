// Package matcher mirrors spacy.matcher.Matcher for the equality-only
// subset — every TokenSpec matches exactly one token position. Quantifier
// operators (?, *, +, !, {n,m}), PhraseMatcher, FUZZY, and Doc.user_data
// extension attrs are deferred (see NOT_YET_PORTED.md).
//
// Typical use:
//
//	m := matcher.New(b.Vocab)
//	m.Add("AI_PATTERN", []matcher.TokenSpec{
//	    {LowerIn: []string{"artificial", "ai"}},
//	    {Lower:   "intelligence"},
//	})
//	for _, hit := range m.Matches(doc) {
//	    fmt.Println(hit.Key, doc.Tokens[hit.Start:hit.End])
//	}
//
// Single-goroutine by gospacy convention: Add mutates the pattern map;
// Matches is safe to call from multiple goroutines once Add is done,
// provided each call uses a per-goroutine Doc (Bundle.Pipe's contract).
package matcher

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/internal/lexflags"
	"github.com/bioshock/gospacy/v3/vocab"
)

// TokenSpec is one token-position predicate. Multiple constraints on the
// same spec AND together. Within a single field, scalar wins over set
// wins over regex; Add validates that conflicting fields aren't set on
// the same spec.
//
// Tri-state int8 fields (IsSpace, IsStop, IsAlpha, IsPunct, IsDigit,
// LikeNum) use: 0 unconstrained, 1 must-be-true, -1 must-be-false.
type TokenSpec struct {
	Orth       string
	OrthIn     []string
	Lower      string
	LowerIn    []string
	LowerRegex *regexp.Regexp
	Tag        string
	TagIn      []string
	TagNotIn   []string
	Pos        string
	PosIn      []string
	Dep        string
	DepIn      []string
	DepNotIn   []string
	Lemma      string
	LemmaIn    []string
	EntType    string
	EntTypeIn  []string

	IsSpace int8
	IsStop  int8
	IsAlpha int8
	IsPunct int8
	IsDigit int8
	LikeNum int8
}

// Match is one hit. Half-open token-index span [Start, End).
type Match struct {
	Key   string
	Start int
	End   int
}

// Matcher applies named patterns to a Doc. One key may have multiple
// alternative patterns; a Doc match against any alternative counts as
// a match under the key.
type Matcher struct {
	vocab    *vocab.Vocab
	patterns map[string][][]TokenSpec
}

// New returns an empty Matcher bound to v. The Vocab is used to resolve
// pattern string values to StringStore hashes; pass the same Vocab the
// Docs you'll Matcher.Matches against were tokenised under.
func New(v *vocab.Vocab) *Matcher {
	return &Matcher{
		vocab:    v,
		patterns: map[string][][]TokenSpec{},
	}
}

// Add registers one or more alternative patterns under key. Each
// alternative is a contiguous token-spec sequence (length >= 1).
// Calling Add twice with the same key appends — does not replace;
// use Remove first to replace.
//
// Returns an error if any spec has conflicting constraints (e.g.
// both Lower and LowerIn set) or if a pattern is empty. Key must
// be non-empty.
func (m *Matcher) Add(key string, patterns ...[]TokenSpec) error {
	if key == "" {
		return fmt.Errorf("matcher.Add: key must be non-empty")
	}
	if len(patterns) == 0 {
		return fmt.Errorf("matcher.Add(%q): at least one pattern required", key)
	}
	for i, pat := range patterns {
		if len(pat) == 0 {
			return fmt.Errorf("matcher.Add(%q): pattern %d is empty", key, i)
		}
		for j, spec := range pat {
			if err := validateSpec(spec); err != nil {
				return fmt.Errorf("matcher.Add(%q): pattern %d spec %d: %w", key, i, j, err)
			}
		}
	}
	m.patterns[key] = append(m.patterns[key], patterns...)
	return nil
}

// validateSpec rejects internally inconsistent specs. For each
// attribute, at most one of {scalar, *In, *NotIn, *Regex} may be set.
// Tri-state flags must be 0 / 1 / -1.
func validateSpec(s TokenSpec) error {
	if s.Orth != "" && len(s.OrthIn) > 0 {
		return fmt.Errorf("cannot set both Orth and OrthIn")
	}
	lowerSet := 0
	if s.Lower != "" {
		lowerSet++
	}
	if len(s.LowerIn) > 0 {
		lowerSet++
	}
	if s.LowerRegex != nil {
		lowerSet++
	}
	if lowerSet > 1 {
		return fmt.Errorf("cannot set more than one of Lower / LowerIn / LowerRegex")
	}
	tagSet := 0
	if s.Tag != "" {
		tagSet++
	}
	if len(s.TagIn) > 0 {
		tagSet++
	}
	if len(s.TagNotIn) > 0 {
		tagSet++
	}
	if tagSet > 1 {
		return fmt.Errorf("cannot set more than one of Tag / TagIn / TagNotIn")
	}
	if s.Pos != "" && len(s.PosIn) > 0 {
		return fmt.Errorf("cannot set both Pos and PosIn")
	}
	depSet := 0
	if s.Dep != "" {
		depSet++
	}
	if len(s.DepIn) > 0 {
		depSet++
	}
	if len(s.DepNotIn) > 0 {
		depSet++
	}
	if depSet > 1 {
		return fmt.Errorf("cannot set more than one of Dep / DepIn / DepNotIn")
	}
	if s.Lemma != "" && len(s.LemmaIn) > 0 {
		return fmt.Errorf("cannot set both Lemma and LemmaIn")
	}
	if s.EntType != "" && len(s.EntTypeIn) > 0 {
		return fmt.Errorf("cannot set both EntType and EntTypeIn")
	}
	for name, v := range map[string]int8{
		"IsSpace": s.IsSpace, "IsStop": s.IsStop, "IsAlpha": s.IsAlpha,
		"IsPunct": s.IsPunct, "IsDigit": s.IsDigit, "LikeNum": s.LikeNum,
	} {
		if v != 0 && v != 1 && v != -1 {
			return fmt.Errorf("%s tri-state must be 0/1/-1, got %d", name, v)
		}
	}
	return nil
}

// Remove deletes all patterns under key. No-op if key absent.
func (m *Matcher) Remove(key string) {
	delete(m.patterns, key)
}

// Matches returns every match in d, in ascending Start order. Overlapping
// matches across different keys are all returned; within a single key,
// overlapping matches are deduplicated longest-first (matches spaCy's
// default for non-quantifier patterns).
func (m *Matcher) Matches(d *doc.Doc) []Match {
	if d == nil || d.NumTokens() == 0 || len(m.patterns) == 0 {
		return nil
	}
	var out []Match
	ss := d.Vocab.StringStore()
	for key, alternatives := range m.patterns {
		var perKey []Match
		for _, spec := range alternatives {
			perKey = append(perKey, scanPattern(d, ss, key, spec)...)
		}
		out = append(out, dedupOverlapsLongestFirst(perKey)...)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Start != out[j].Start {
			return out[i].Start < out[j].Start
		}
		if out[i].End != out[j].End {
			return out[i].End < out[j].End
		}
		return out[i].Key < out[j].Key
	})
	return out
}

// scanPattern walks d.Tokens looking for contiguous runs that satisfy
// every TokenSpec in spec. Returns one Match per hit; overlap dedup is
// applied later, per key.
func scanPattern(d *doc.Doc, ss *vocab.StringStore, key string, spec []TokenSpec) []Match {
	if len(spec) == 0 {
		return nil
	}
	var out []Match
	n := d.NumTokens()
	for i := 0; i <= n-len(spec); i++ {
		ok := true
		for j := 0; j < len(spec); j++ {
			if !testSpec(d, ss, &d.Tokens[i+j], &spec[j]) {
				ok = false
				break
			}
		}
		if ok {
			out = append(out, Match{Key: key, Start: i, End: i + len(spec)})
		}
	}
	return out
}

// testSpec evaluates one TokenSpec against one Token. Returns false on
// the first failing constraint.
func testSpec(d *doc.Doc, ss *vocab.StringStore, tok *doc.Token, spec *TokenSpec) bool {
	// Orth equality / set membership.
	if spec.Orth != "" {
		h, present := ss.Get(spec.Orth)
		if !present || tok.Orth != h {
			return false
		}
	}
	if len(spec.OrthIn) > 0 && !hashIn(ss, tok.Orth, spec.OrthIn) {
		return false
	}
	// Lower equality / set / regex.
	if spec.Lower != "" {
		h, present := ss.Get(spec.Lower)
		if !present || tok.Lower != h {
			return false
		}
	}
	if len(spec.LowerIn) > 0 && !hashIn(ss, tok.Lower, spec.LowerIn) {
		return false
	}
	if spec.LowerRegex != nil && !spec.LowerRegex.MatchString(strings.ToLower(tok.Text)) {
		return false
	}
	// Tag equality / set / negation.
	if spec.Tag != "" {
		h, present := ss.Get(spec.Tag)
		if !present || tok.Tag != h {
			return false
		}
	}
	if len(spec.TagIn) > 0 && !hashIn(ss, tok.Tag, spec.TagIn) {
		return false
	}
	if len(spec.TagNotIn) > 0 && hashInWithEmptySentinel(ss, tok.Tag, spec.TagNotIn) {
		return false
	}
	// Pos equality / set.
	if spec.Pos != "" {
		h, present := ss.Get(spec.Pos)
		if !present || tok.POS != h {
			return false
		}
	}
	if len(spec.PosIn) > 0 && !hashIn(ss, tok.POS, spec.PosIn) {
		return false
	}
	// Dep equality / set / negation.
	if spec.Dep != "" {
		h, present := ss.Get(spec.Dep)
		if !present || tok.Dep != h {
			return false
		}
	}
	if len(spec.DepIn) > 0 && !hashIn(ss, tok.Dep, spec.DepIn) {
		return false
	}
	if len(spec.DepNotIn) > 0 && hashInWithEmptySentinel(ss, tok.Dep, spec.DepNotIn) {
		return false
	}
	// Lemma equality / set.
	if spec.Lemma != "" {
		h, present := ss.Get(spec.Lemma)
		if !present || tok.Lemma != h {
			return false
		}
	}
	if len(spec.LemmaIn) > 0 && !hashIn(ss, tok.Lemma, spec.LemmaIn) {
		return false
	}
	// EntType equality / set.
	if spec.EntType != "" {
		h, present := ss.Get(spec.EntType)
		if !present || tok.EntType != h {
			return false
		}
	}
	if len(spec.EntTypeIn) > 0 && !hashIn(ss, tok.EntType, spec.EntTypeIn) {
		return false
	}
	// Tri-state boolean flags.
	if spec.IsSpace != 0 {
		got := isAllSpace(tok.Text)
		if (spec.IsSpace == 1) != got {
			return false
		}
	}
	if spec.IsStop != 0 {
		got := tok.IsStop(d.Vocab)
		if (spec.IsStop == 1) != got {
			return false
		}
	}
	if spec.IsAlpha != 0 {
		got := lexflags.IsAlpha(tok.Text)
		if (spec.IsAlpha == 1) != got {
			return false
		}
	}
	if spec.IsPunct != 0 {
		got := lexflags.IsPunct(tok.Text)
		if (spec.IsPunct == 1) != got {
			return false
		}
	}
	if spec.IsDigit != 0 {
		got := lexflags.IsDigit(tok.Text)
		if (spec.IsDigit == 1) != got {
			return false
		}
	}
	if spec.LikeNum != 0 {
		got := lexflags.LikeNum(tok.Text)
		if (spec.LikeNum == 1) != got {
			return false
		}
	}
	return true
}

// hashIn — true when h equals StringStore.Get(s) for any s in set.
// Set strings that aren't interned are silently skipped (no auto-Add).
func hashIn(ss *vocab.StringStore, h uint64, set []string) bool {
	for _, s := range set {
		if sh, ok := ss.Get(s); ok && sh == h {
			return true
		}
	}
	return false
}

// hashInWithEmptySentinel — true when h matches any entry in set;
// the "" sentinel matches when h is 0 (i.e. attribute unset). Mirrors
// AttributeRuler's TagNotIn semantics.
func hashInWithEmptySentinel(ss *vocab.StringStore, h uint64, set []string) bool {
	for _, s := range set {
		if s == "" {
			if h == 0 {
				return true
			}
			continue
		}
		if sh, ok := ss.Get(s); ok && sh == h {
			return true
		}
	}
	return false
}

// isAllSpace — true iff every rune in s is unicode whitespace.
// Mirrors pipeline/attributeruler.go:tokenIsAllSpace. Empty matches
// spaCy's is_space convention (returns true for "").
func isAllSpace(s string) bool {
	if s == "" {
		return true
	}
	for _, r := range s {
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return true
}

// dedupOverlapsLongestFirst removes overlapping matches within a single
// key, keeping the longer span when two overlap. Matches must all share
// the same Key. Stable on tie (earlier-start wins).
func dedupOverlapsLongestFirst(matches []Match) []Match {
	if len(matches) <= 1 {
		return matches
	}
	// Sort by length desc, then by start asc (stable).
	sort.SliceStable(matches, func(i, j int) bool {
		li := matches[i].End - matches[i].Start
		lj := matches[j].End - matches[j].Start
		if li != lj {
			return li > lj
		}
		return matches[i].Start < matches[j].Start
	})
	var kept []Match
	for _, m := range matches {
		overlap := false
		for _, k := range kept {
			if m.Start < k.End && k.Start < m.End {
				overlap = true
				break
			}
		}
		if !overlap {
			kept = append(kept, m)
		}
	}
	return kept
}
