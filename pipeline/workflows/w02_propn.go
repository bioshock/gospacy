package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w02Propn lists the surface text of every token with POS == PROPN, in
// token order.
func w02Propn(d *doc.Doc, ss *vocab.StringStore) any {
	out := []string{}
	for _, t := range d.Tokens {
		pos, _ := ss.Lookup(t.POS)
		if pos == "PROPN" {
			out = append(out, t.Text)
		}
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w02_propn",
		Run:        w02Propn,
		GoldenPath: "../../testdata/golden/workflows/w02_propn.json",
	})
}
