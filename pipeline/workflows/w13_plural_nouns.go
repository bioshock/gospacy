package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w13PluralNouns lists token text for every NOUN with morphology
// Number=Plur.
func w13PluralNouns(d *doc.Doc, ss *vocab.StringStore) any {
	out := []string{}
	for _, t := range d.Tokens {
		pos, _ := ss.Lookup(t.POS)
		if pos != "NOUN" {
			continue
		}
		if !morphHas(t.Morph, "Number", "Plur") {
			continue
		}
		out = append(out, t.Text)
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w13_plural_nouns",
		Run:        w13PluralNouns,
		GoldenPath: "../../testdata/golden/workflows/w13_plural_nouns.json",
	})
}
