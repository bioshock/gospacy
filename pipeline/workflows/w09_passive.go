package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w09Passive lists every token with dep_ == "nsubjpass" plus the head
// token's text.
func w09Passive(d *doc.Doc, ss *vocab.StringStore) any {
	out := []map[string]string{}
	for _, t := range d.Tokens {
		dep, _ := ss.Lookup(t.Dep)
		if dep != "nsubjpass" {
			continue
		}
		out = append(out, map[string]string{
			"text": t.Text,
			"head": d.Tokens[t.Head].Text,
		})
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w09_passive",
		Run:        w09Passive,
		GoldenPath: "../../testdata/golden/workflows/w09_passive.json",
	})
}
