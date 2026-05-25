package pipeline

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// AttributeRuler writes Token.POS, Token.Morph (and optionally Tag) by
// matching token-level patterns and applying their attrs. Mirrors
// spacy.pipeline.attributeruler.AttributeRuler in the minimal subset
// en_core_web_sm exercises: single-token TAG/ORTH patterns.
type AttributeRuler struct {
	patterns []arPattern
	warnOnce sync.Once
}

// NewAttributeRuler loads the patterns file from <bundle>/attribute_ruler/patterns.
func NewAttributeRuler(b BundleSource) (*AttributeRuler, error) {
	path := filepath.Join(b.BundlePath(), "attribute_ruler", "patterns")
	pats, err := loadAttributeRulerPatterns(path)
	if err != nil {
		return nil, fmt.Errorf("NewAttributeRuler: %w", err)
	}
	// Intern every emitted attribute string and every token-spec string so
	// Apply / matchPattern can resolve hashes via ss.Get without re-adding
	// per token. Token-spec strings (LOWER/ORTH/TAG values) must be present
	// in the store before matchPattern runs, otherwise ss.Get returns
	// (0, false) and the pattern silently fails to match.
	ss := b.BundleVocab().StringStore()
	for _, p := range pats {
		if p.Attrs.POS != "" {
			ss.Add(p.Attrs.POS)
		}
		if p.Attrs.Tag != "" {
			ss.Add(p.Attrs.Tag)
		}
		if p.Attrs.Lemma != "" {
			ss.Add(p.Attrs.Lemma)
		}
		for _, spec := range p.TokenSpecs {
			if spec.Tag != "" {
				ss.Add(spec.Tag)
			}
			if spec.Orth != "" {
				ss.Add(spec.Orth)
			}
			if spec.Lower != "" {
				ss.Add(spec.Lower)
			}
			if spec.Dep != "" {
				ss.Add(spec.Dep)
			}
			for _, s := range spec.TagIn {
				ss.Add(s)
			}
			for _, s := range spec.OrthIn {
				ss.Add(s)
			}
			for _, s := range spec.LowerIn {
				ss.Add(s)
			}
			for _, s := range spec.TagNotIn {
				if s != "" {
					ss.Add(s)
				}
			}
			for _, s := range spec.DepIn {
				ss.Add(s)
			}
			for _, s := range spec.DepNotIn {
				if s != "" {
					ss.Add(s)
				}
			}
		}
	}
	return &AttributeRuler{patterns: pats}, nil
}

// Apply runs every pattern over the doc. Patterns are applied in registration
// order; the LAST winning pattern's attrs survive on each token.
func (a *AttributeRuler) Apply(d *doc.Doc) error {
	if d.NumTokens() == 0 {
		return nil
	}
	ss := d.Vocab.StringStore()
	for _, p := range a.patterns {
		if p.Unsupported {
			a.warnOnce.Do(func() {
				fmt.Fprintf(stderrSink(), "pipeline.AttributeRuler: pattern with unsupported key skipped (will not match)\n")
			})
			continue
		}
		matches := matchPattern(d, p)
		for _, start := range matches {
			tokIdx := start + p.Index
			if tokIdx < 0 || tokIdx >= d.NumTokens() {
				continue
			}
			tok := &d.Tokens[tokIdx]
			if p.Attrs.POS != "" {
				if h, ok := ss.Get(p.Attrs.POS); ok {
					tok.POS = h
				}
			}
			if p.Attrs.Tag != "" {
				if h, ok := ss.Get(p.Attrs.Tag); ok {
					tok.Tag = h
				}
			}
			if p.Attrs.Morph != "" {
				// spaCy stores MORPH="_" as the "no morphology" sentinel; the
				// Python `str(Token.morph)` for that case is "". Translate at
				// the write site so callers see the round-tripped Python
				// representation directly.
				if p.Attrs.Morph == "_" {
					tok.Morph = ""
				} else {
					tok.Morph = p.Attrs.Morph
				}
			}
			if p.Attrs.Lemma != "" {
				// Patterns carrying a LEMMA attr write Token.Lemma. The
				// downstream Lemmatizer is configured with overwrite=false in
				// en_core_web_sm, so the lemma we set here survives the
				// lemmatizer pass — mirroring spaCy's pipeline order
				// (attribute_ruler before lemmatizer).
				if h, ok := ss.Get(p.Attrs.Lemma); ok {
					tok.Lemma = h
				}
			}
		}
	}
	return nil
}

// matchPattern returns the start indices in d where pattern p matches. The
// matcher implements TAG / ORTH / LOWER / DEP equality (scalar field),
// set-membership ({"IN": [...]}, normalized into the *In slices by the
// loader), negated set-membership ({"NOT_IN": [...]} for TAG / DEP),
// LOWER regex matching ({"REGEX": "..."}) against strings.ToLower(Text),
// and IS_SPACE (boolean over unicode.IsSpace of Token.Text) on consecutive
// tokens. Unsupported keys are reported via arPattern.Unsupported and
// never reach matchPattern.
func matchPattern(d *doc.Doc, p arPattern) []int {
	if len(p.TokenSpecs) == 0 {
		return nil
	}
	ss := d.Vocab.StringStore()
	var out []int
	specRunes := len(p.TokenSpecs)
	for i := 0; i <= d.NumTokens()-specRunes; i++ {
		ok := true
		for j, spec := range p.TokenSpecs {
			tok := d.Tokens[i+j]
			if spec.Tag != "" {
				h, present := ss.Get(spec.Tag)
				if !present || tok.Tag != h {
					ok = false
					break
				}
			}
			if len(spec.TagIn) > 0 && !hashInSet(ss, tok.Tag, spec.TagIn) {
				ok = false
				break
			}
			if spec.Orth != "" {
				h, present := ss.Get(spec.Orth)
				if !present || tok.Orth != h {
					ok = false
					break
				}
			}
			if len(spec.OrthIn) > 0 && !hashInSet(ss, tok.Orth, spec.OrthIn) {
				ok = false
				break
			}
			if spec.Lower != "" {
				h, present := ss.Get(spec.Lower)
				if !present || tok.Lower != h {
					ok = false
					break
				}
			}
			if len(spec.LowerIn) > 0 && !hashInSet(ss, tok.Lower, spec.LowerIn) {
				ok = false
				break
			}
			if spec.LowerRegex != nil && !spec.LowerRegex.MatchString(strings.ToLower(tok.Text)) {
				ok = false
				break
			}
			if spec.Dep != "" {
				h, present := ss.Get(spec.Dep)
				if !present || tok.Dep != h {
					ok = false
					break
				}
			}
			if len(spec.DepIn) > 0 && !hashInSet(ss, tok.Dep, spec.DepIn) {
				ok = false
				break
			}
			if len(spec.TagNotIn) > 0 {
				blocked := false
				for _, s := range spec.TagNotIn {
					if s == "" {
						// "" sentinel: tok must have a non-zero Tag hash.
						if tok.Tag == 0 {
							blocked = true
							break
						}
						continue
					}
					if h, present := ss.Get(s); present && tok.Tag == h {
						blocked = true
						break
					}
				}
				if blocked {
					ok = false
					break
				}
			}
			if len(spec.DepNotIn) > 0 {
				blocked := false
				for _, s := range spec.DepNotIn {
					if s == "" {
						if tok.Dep == 0 {
							blocked = true
							break
						}
						continue
					}
					if h, present := ss.Get(s); present && tok.Dep == h {
						blocked = true
						break
					}
				}
				if blocked {
					ok = false
					break
				}
			}
			if spec.IsSpace != 0 {
				isSpace := tokenIsAllSpace(tok.Text)
				if (spec.IsSpace == 1 && !isSpace) || (spec.IsSpace == -1 && isSpace) {
					ok = false
					break
				}
			}
		}
		if ok {
			out = append(out, i)
		}
	}
	return out
}

// tokenIsAllSpace reports whether s consists entirely of unicode whitespace.
// Mirrors spaCy's Token.is_space which is true iff every char in the surface
// form is whitespace (or the string is empty). Used by IS_SPACE patterns.
func tokenIsAllSpace(s string) bool {
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

// hashInSet returns true when h equals the StringStore hash of any string in
// set. Set strings are looked up via ss.Get (no auto-Add); NewAttributeRuler
// interns them at construction, so by the time this runs they are present.
func hashInSet(ss *vocab.StringStore, h uint64, set []string) bool {
	for _, s := range set {
		sh, ok := ss.Get(s)
		if ok && sh == h {
			return true
		}
	}
	return false
}

// stderrSink is a seam so tests can redirect AttributeRuler warnings without
// touching os.Stderr globally.
var stderrSinkFn = func() interface {
	Write([]byte) (int, error)
} {
	return os.Stderr
}

func stderrSink() interface{ Write([]byte) (int, error) } { return stderrSinkFn() }
