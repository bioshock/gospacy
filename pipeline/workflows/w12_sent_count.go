package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w12SentCount returns the number of sentences in the Doc.
func w12SentCount(d *doc.Doc, ss *vocab.StringStore) any {
	return len(d.Sents())
}

func init() {
	register(Workflow{
		Name:       "w12_sent_count",
		Run:        w12SentCount,
		GoldenPath: "../../testdata/golden/workflows/w12_sent_count.json",
	})
}
