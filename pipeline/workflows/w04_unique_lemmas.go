package workflows

import (
	"sort"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w04UniqueLemmas returns the sorted set of distinct Lemma strings.
func w04UniqueLemmas(d *doc.Doc, ss *vocab.StringStore) any {
	seen := map[string]struct{}{}
	for _, t := range d.Tokens {
		lemma, _ := ss.Lookup(t.Lemma)
		seen[lemma] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func init() {
	register(Workflow{
		Name:       "w04_unique_lemmas",
		Run:        w04UniqueLemmas,
		GoldenPath: "../../testdata/golden/workflows/w04_unique_lemmas.json",
	})
}
