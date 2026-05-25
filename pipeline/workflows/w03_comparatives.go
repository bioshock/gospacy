package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

var comparativeTags = map[string]struct{}{
	"JJR": {},
	"JJS": {},
	"RBR": {},
	"RBS": {},
}

// w03Comparatives lists every token with a comparative/superlative Penn tag
// (JJR/JJS/RBR/RBS) as {text, tag}.
func w03Comparatives(d *doc.Doc, ss *vocab.StringStore) any {
	out := []map[string]string{}
	for _, t := range d.Tokens {
		tag, _ := ss.Lookup(t.Tag)
		if _, ok := comparativeTags[tag]; !ok {
			continue
		}
		out = append(out, map[string]string{"text": t.Text, "tag": tag})
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w03_comparatives",
		Run:        w03Comparatives,
		GoldenPath: "../../testdata/golden/workflows/w03_comparatives.json",
	})
}
