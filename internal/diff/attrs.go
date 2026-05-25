package diff

import "fmt"

// AttrKind names which token attribute disagreed.
type AttrKind int

const (
	AttrOrth  AttrKind = iota + 1 // token text mismatch (token-alignment issue)
	AttrTag                       // Penn-Treebank POS tag mismatch
	AttrPOS                       // Universal Dependencies coarse POS mismatch
	AttrMorph                     // morphological features string mismatch
	AttrLemma                     // lemma mismatch
)

// String returns a human-readable name for the AttrKind constant.
func (k AttrKind) String() string {
	switch k {
	case AttrOrth:
		return "orth"
	case AttrTag:
		return "tag"
	case AttrPOS:
		return "pos"
	case AttrMorph:
		return "morph"
	case AttrLemma:
		return "lemma"
	default:
		return fmt.Sprintf("unknown(%d)", int(k))
	}
}

// AttrDisagreement records one per-token attribute mismatch between expected
// and actual output, naming the attribute kind and the differing values.
type AttrDisagreement struct {
	Attr  AttrKind
	Index int
	Want  string
	Got   string
}

// AttrReport summarises attribute-level disagreements. A LengthDisagreement
// at index -1 indicates the token sequences have different lengths.
type AttrReport struct {
	LengthDisagreement *Disagreement
	AttrDisagreements  []AttrDisagreement
}

// Equal reports whether the comparison found no disagreements.
func (r AttrReport) Equal() bool {
	return r.LengthDisagreement == nil && len(r.AttrDisagreements) == 0
}

// CompareAttrs diffs two TokenAttr slices and returns an AttrReport.
// On length mismatch it stops without per-token comparison.
func CompareAttrs(want, got []TokenAttr) AttrReport {
	var r AttrReport
	if len(want) != len(got) {
		r.LengthDisagreement = &Disagreement{
			Kind: DisagreeLength, Index: -1,
			Want: fmt.Sprintf("%d tokens", len(want)),
			Got:  fmt.Sprintf("%d tokens", len(got)),
		}
		return r
	}
	for i := range want {
		w, g := want[i], got[i]
		if w.Orth != g.Orth {
			r.AttrDisagreements = append(r.AttrDisagreements,
				AttrDisagreement{Attr: AttrOrth, Index: i, Want: w.Orth, Got: g.Orth})
			continue // wrong orth invalidates the rest at this index
		}
		if w.Tag != g.Tag {
			r.AttrDisagreements = append(r.AttrDisagreements,
				AttrDisagreement{Attr: AttrTag, Index: i, Want: w.Tag, Got: g.Tag})
		}
		if w.POS != g.POS {
			r.AttrDisagreements = append(r.AttrDisagreements,
				AttrDisagreement{Attr: AttrPOS, Index: i, Want: w.POS, Got: g.POS})
		}
		if w.Morph != g.Morph {
			r.AttrDisagreements = append(r.AttrDisagreements,
				AttrDisagreement{Attr: AttrMorph, Index: i, Want: w.Morph, Got: g.Morph})
		}
		if w.Lemma != g.Lemma {
			r.AttrDisagreements = append(r.AttrDisagreements,
				AttrDisagreement{Attr: AttrLemma, Index: i, Want: w.Lemma, Got: g.Lemma})
		}
	}
	return r
}
