package workflows

import (
	"strings"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w16Questions reports one bool per sentence: true if the first token's
// fine-grained tag starts with "W" (WDT/WP/WP$/WRB) OR the last token's
// text is "?".
func w16Questions(d *doc.Doc, ss *vocab.StringStore) any {
	sents := d.Sents()
	out := make([]bool, 0, len(sents))
	for _, s := range sents {
		if s.Len() == 0 {
			out = append(out, false)
			continue
		}
		firstTag, _ := ss.Lookup(d.Tokens[s.Start].Tag)
		lastText := d.Tokens[s.End-1].Text
		isQ := strings.HasPrefix(firstTag, "W") || lastText == "?"
		out = append(out, isQ)
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w16_questions",
		Run:        w16Questions,
		GoldenPath: "../../testdata/golden/workflows/w16_questions.json",
	})
}
