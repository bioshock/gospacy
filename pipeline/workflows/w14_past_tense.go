package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w14PastTense lists VERBs with morphology Tense=Past.
func w14PastTense(d *doc.Doc, ss *vocab.StringStore) any {
	out := []string{}
	for _, t := range d.Tokens {
		pos, _ := ss.Lookup(t.POS)
		if pos != "VERB" {
			continue
		}
		if !morphHas(t.Morph, "Tense", "Past") {
			continue
		}
		out = append(out, t.Text)
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w14_past_tense",
		Run:        w14PastTense,
		GoldenPath: "../../testdata/golden/workflows/w14_past_tense.json",
	})
}
