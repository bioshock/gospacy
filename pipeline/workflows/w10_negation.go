package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w10Negation emits {neg, head, head_pos} for every token with dep_ ==
// "neg".
func w10Negation(d *doc.Doc, ss *vocab.StringStore) any {
	out := []map[string]string{}
	for _, t := range d.Tokens {
		dep, _ := ss.Lookup(t.Dep)
		if dep != "neg" {
			continue
		}
		headPOS, _ := ss.Lookup(d.Tokens[t.Head].POS)
		out = append(out, map[string]string{
			"neg":      t.Text,
			"head":     d.Tokens[t.Head].Text,
			"head_pos": headPOS,
		})
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w10_negation",
		Run:        w10Negation,
		GoldenPath: "../../testdata/golden/workflows/w10_negation.json",
	})
}
