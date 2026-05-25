package doc

import "strings"

// Span is a half-open [Start,End) slice over Doc.Tokens. Mirrors
// `spacy.tokens.span.Span`. Cheap to construct; does not copy tokens.
type Span struct {
	Doc        *Doc
	Start, End int
}

// Len returns End - Start.
func (s Span) Len() int { return s.End - s.Start }

// Text returns the concatenated TextWithWS of the spanned tokens, with the
// trailing whitespace of the final token trimmed. Mirrors `Span.text`.
func (s Span) Text() string {
	if s.Len() <= 0 {
		return ""
	}
	var b strings.Builder
	for i := s.Start; i < s.End; i++ {
		if i == s.End-1 {
			b.WriteString(s.Doc.Tokens[i].Text)
		} else {
			b.WriteString(s.Doc.Tokens[i].TextWithWS())
		}
	}
	return b.String()
}

// At returns a pointer to the token at offset i within the span. Negative i
// indexes from the end (-1 = last). Panics on out-of-range, matching slice
// semantics (the caller is responsible — AttributeRuler validates first).
func (s Span) At(i int) *Token {
	if i < 0 {
		i = s.Len() + i
	}
	return &s.Doc.Tokens[s.Start+i]
}
