package doc

import "fmt"

// Sents partitions the Doc into sentence Spans using the parser-set
// SentStart markers. Each Span is half-open [start,end). The first sentence
// always starts at index 0; subsequent sentences start at the next i where
// Tokens[i].SentStart == 1.
//
// Panics if any token has SentStart == -1 (unknown). The caller must have
// run the parser (or a sentencizer) before calling Sents; this matches
// spaCy's behaviour of raising ValueError when sentence boundaries are
// missing. Empty Docs return nil.
func (d *Doc) Sents() []Span {
	if len(d.Tokens) == 0 {
		return nil
	}
	for i := range d.Tokens {
		if d.Tokens[i].SentStart == -1 {
			panic(fmt.Sprintf("doc.Sents: token %d has SentStart == -1; run the parser first", i))
		}
	}
	var out []Span
	start := 0
	for i := 1; i < len(d.Tokens); i++ {
		if d.Tokens[i].SentStart == 1 {
			out = append(out, Span{Doc: d, Start: start, End: i})
			start = i
		}
	}
	out = append(out, Span{Doc: d, Start: start, End: len(d.Tokens)})
	return out
}
