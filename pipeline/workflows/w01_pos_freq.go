package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w01PosFreq counts each POS tag in token order and returns a map keyed by
// POS string (e.g. "NOUN"). Matches dump_w01_pos_freq.py exactly.
func w01PosFreq(d *doc.Doc, ss *vocab.StringStore) any {
	out := map[string]int{}
	for _, t := range d.Tokens {
		pos, _ := ss.Lookup(t.POS)
		out[pos]++
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w01_pos_freq",
		Run:        w01PosFreq,
		GoldenPath: "../../testdata/golden/workflows/w01_pos_freq.json",
	})
}
