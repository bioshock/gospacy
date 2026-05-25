package doc

// nounChunkDepLabels is the np_deps list from
// spacy/lang/en/syntax_iterators.py (10 labels including "ROOT").
var nounChunkDepLabels = []string{
	"oprd",
	"nsubj",
	"dobj",
	"nsubjpass",
	"pcomp",
	"pobj",
	"dative",
	"appos",
	"attr",
	"ROOT",
}

// NounChunks returns base noun-phrase spans, ported from
// spacy.lang.en.syntax_iterators.noun_chunks. Walks tokens left-to-right;
// for each token whose POS is NOUN/PROPN/PRON, emits the [left_edge,i+1)
// span when the dep is in np_deps OR when the dep is "conj" and the
// resolved head's dep is in np_deps. Skips nested chunks by tracking the
// previous chunk's end and rejecting tokens whose left_edge lies at or
// before that boundary.
//
// English-only. Caller must have run the parser (sentences without DEP
// labels yield an empty result rather than panicking; spaCy raises
// ValueError in that case but we can't tell "no chunks" from "no parse"
// without a separate has-annotation flag, so we silently return nil).
func (d *Doc) NounChunks() []Span {
	if len(d.Tokens) == 0 {
		return nil
	}
	ss := d.Vocab.StringStore()
	npDeps := make(map[uint64]struct{}, len(nounChunkDepLabels))
	for _, lab := range nounChunkDepLabels {
		npDeps[ss.Hash(lab)] = struct{}{}
	}
	conjHash := ss.Hash("conj")
	nounHash := ss.Hash("NOUN")
	propnHash := ss.Hash("PROPN")
	pronHash := ss.Hash("PRON")

	prevEnd := -1
	var out []Span
	for i := range d.Tokens {
		pos := d.Tokens[i].POS
		if pos != nounHash && pos != propnHash && pos != pronHash {
			continue
		}
		// left_edge = min index in the token's subtree.
		sub := SubtreeOf(d, i)
		if len(sub) == 0 {
			continue
		}
		leftEdge := sub[0]
		if leftEdge <= prevEnd {
			continue
		}
		dep := d.Tokens[i].Dep
		if _, ok := npDeps[dep]; ok {
			prevEnd = i
			out = append(out, Span{Doc: d, Start: leftEdge, End: i + 1})
			continue
		}
		if dep == conjHash {
			head := d.Tokens[i].Head
			// Climb conj chain while head.dep == conj and head.head.i < head.i.
			for d.Tokens[head].Dep == conjHash && d.Tokens[head].Head < head {
				head = d.Tokens[head].Head
			}
			if _, ok := npDeps[d.Tokens[head].Dep]; ok {
				prevEnd = i
				out = append(out, Span{Doc: d, Start: leftEdge, End: i + 1})
			}
		}
	}
	return out
}

