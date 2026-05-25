package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w18Entities walks the per-token EntIOB stream and emits one entry per
// entity span. Mirrors Python:
//
//	[{"start": e.start, "end": e.end, "label": e.label_, "text": e.text}
//	 for e in doc.ents]
//
// IOB transition rules: B starts a new entity; I/L continues (we collapse
// L to I in writeback, so I covers both); anything else (O / missing) ends
// the current entity.
func w18Entities(d *doc.Doc, ss *vocab.StringStore) any {
	out := make([]map[string]any, 0)
	var inEnt bool
	var entStart int
	var entLabel string
	flush := func(end int) {
		if !inEnt {
			return
		}
		out = append(out, map[string]any{
			"start": entStart,
			"end":   end,
			"label": entLabel,
			"text":  spanText(d, entStart, end),
		})
		inEnt = false
		entLabel = ""
	}
	for i := range d.Tokens {
		switch d.Tokens[i].EntIOB {
		case 3: // B
			flush(i) // close any open ent before starting a new one
			inEnt = true
			entStart = i
			entLabel, _ = ss.Lookup(d.Tokens[i].EntType)
		case 1: // I
			// continues the open entity; nothing to do.
		default: // O (2), missing (0)
			flush(i)
		}
	}
	flush(len(d.Tokens))
	return out
}

// spanText reconstructs the surface text for tokens [start, end), reusing
// each token's trailing whitespace so multi-token entities render exactly
// like spaCy's Span.text.
func spanText(d *doc.Doc, start, end int) string {
	if start < 0 || end > len(d.Tokens) || start >= end {
		return ""
	}
	out := ""
	for i := start; i < end; i++ {
		out += d.Tokens[i].Text
		// trailing whitespace except on the last token of the span.
		if i < end-1 {
			out += d.Tokens[i].Whitespace
		}
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w18_entities",
		Run:        w18Entities,
		GoldenPath: "../../testdata/golden/workflows/w18_entities.json",
	})
}
