package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w15NounChunks emits {start, end, text} for each base noun phrase via
// doc.NounChunks (port of spacy/lang/en/syntax_iterators.noun_chunks).
func w15NounChunks(d *doc.Doc, ss *vocab.StringStore) any {
	chunks := d.NounChunks()
	out := make([]map[string]any, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, map[string]any{
			"start": c.Start,
			"end":   c.End,
			"text":  c.Text(),
		})
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w15_noun_chunks",
		Run:        w15NounChunks,
		GoldenPath: "../../testdata/golden/workflows/w15_noun_chunks.json",
	})
}
