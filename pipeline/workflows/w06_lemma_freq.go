package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w06LemmaFreq counts lemmas excluding stop words and punctuation.
func w06LemmaFreq(d *doc.Doc, ss *vocab.StringStore) any {
	v := vocabForDoc(d)
	out := map[string]int{}
	for _, t := range d.Tokens {
		if t.IsStop(v) || isPunctToken(t, v) {
			continue
		}
		lemma, _ := ss.Lookup(t.Lemma)
		out[lemma]++
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w06_lemma_freq",
		Run:        w06LemmaFreq,
		GoldenPath: "../../testdata/golden/workflows/w06_lemma_freq.json",
	})
}
