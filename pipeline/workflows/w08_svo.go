package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// w08SVO emits [subj_text, verb_text, obj_text] triples for every VERB
// token whose direct children include both an nsubj and a dobj. Mirrors
// the dump_w08_svo.py loop exactly: first nsubj child wins for subj, first
// dobj child wins for obj.
func w08SVO(d *doc.Doc, ss *vocab.StringStore) any {
	out := [][]string{}
	for i, t := range d.Tokens {
		pos, _ := ss.Lookup(t.POS)
		if pos != "VERB" {
			continue
		}
		var subj, obj string
		var haveSubj, haveObj bool
		for _, ci := range doc.ChildrenOf(d, i) {
			dep, _ := ss.Lookup(d.Tokens[ci].Dep)
			if dep == "nsubj" && !haveSubj {
				subj = d.Tokens[ci].Text
				haveSubj = true
			} else if dep == "dobj" && !haveObj {
				obj = d.Tokens[ci].Text
				haveObj = true
			}
		}
		if haveSubj && haveObj {
			out = append(out, []string{subj, t.Text, obj})
		}
	}
	return out
}

func init() {
	register(Workflow{
		Name:       "w08_svo",
		Run:        w08SVO,
		GoldenPath: "../../testdata/golden/workflows/w08_svo.json",
	})
}
