package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w05ContentWords lists lemmas of tokens that are neither stop words nor
// punctuation, in token order.
func w05ContentWords(d *doc.Doc, ss *vocab.StringStore) any {
	v := vocabForDoc(d)
	out := []string{}
	for _, t := range d.Tokens {
		if t.IsStop(v) {
			continue
		}
		if isPunctToken(t, v) {
			continue
		}
		lemma, _ := ss.Lookup(t.Lemma)
		out = append(out, lemma)
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w05_content_words",
		Run:        w05ContentWords,
		GoldenPath: "../../testdata/golden/workflows/w05_content_words.json",
	})
}
