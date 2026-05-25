package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w11SentSegmentation returns each sentence's text.
func w11SentSegmentation(d *doc.Doc, ss *vocab.StringStore) any {
	sents := d.Sents()
	out := make([]string, 0, len(sents))
	for _, s := range sents {
		out = append(out, s.Text())
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w11_sent_segmentation",
		Run:        w11SentSegmentation,
		GoldenPath: "../../testdata/golden/workflows/w11_sent_segmentation.json",
	})
}
