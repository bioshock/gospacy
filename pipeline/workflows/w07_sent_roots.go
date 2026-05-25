package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w07SentRoots lists every token with dep_ == "ROOT" in token order.
func w07SentRoots(d *doc.Doc, ss *vocab.StringStore) any {
	out := []map[string]string{}
	for _, t := range d.Tokens {
		dep, _ := ss.Lookup(t.Dep)
		if dep != "ROOT" {
			continue
		}
		pos, _ := ss.Lookup(t.POS)
		out = append(out, map[string]string{
			"text": t.Text,
			"dep":  dep,
			"pos":  pos,
		})
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w07_sent_roots",
		Run:        w07SentRoots,
		GoldenPath: "../../testdata/golden/workflows/w07_sent_roots.json",
	})
}
