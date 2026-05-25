package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w17Keywords counts NOUN/PROPN lemmas that aren't stop words.
func w17Keywords(d *doc.Doc, ss *vocab.StringStore) any {
	v := vocabForDoc(d)
	out := map[string]int{}
	for _, t := range d.Tokens {
		pos, _ := ss.Lookup(t.POS)
		if pos != "NOUN" && pos != "PROPN" {
			continue
		}
		if t.IsStop(v) {
			continue
		}
		lemma, _ := ss.Lookup(t.Lemma)
		out[lemma]++
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w17_keywords",
		Run:        w17Keywords,
		GoldenPath: "../../testdata/golden/workflows/w17_keywords.json",
	})
}
