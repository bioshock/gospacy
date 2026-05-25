package diff

import "fmt"

// ArcReport summarises dep-parse disagreement plus standard parser metrics.
type ArcReport struct {
	LengthDisagreement *Disagreement
	ArcDisagreements   []ArcDisagreement
	// UAS = unlabeled attachment score = (# tokens with correct head) / N
	// LAS = labeled attachment score   = (# tokens with correct head AND label) / N
	UAS float64
	LAS float64
}

// ArcDisagreement records one token's head or label mismatch in a dep-parse
// comparison, with both the expected and actual head index and label.
type ArcDisagreement struct {
	Index    int    // token index
	WantHead int
	GotHead  int
	WantDep  string
	GotDep   string
}

// Equal reports whether the comparison found no disagreements (UAS = LAS = 1.0).
func (r ArcReport) Equal() bool {
	return r.LengthDisagreement == nil && len(r.ArcDisagreements) == 0
}

// CompareArcs diffs two arc slices. Token indices are assumed to align (the
// caller has already verified token sequences match via CompareTokens).
func CompareArcs(want, got []Arc) ArcReport {
	var r ArcReport
	if len(want) != len(got) {
		r.LengthDisagreement = &Disagreement{
			Kind: DisagreeLength, Index: -1,
			Want: fmt.Sprintf("%d arcs", len(want)),
			Got:  fmt.Sprintf("%d arcs", len(got)),
		}
		return r
	}
	if len(want) == 0 {
		// no tokens, no scores; treat as fully equal
		return r
	}
	uasHits, lasHits := 0, 0
	for i := range want {
		w, g := want[i], got[i]
		headEqual := w.Head == g.Head
		depEqual := w.Dep == g.Dep
		if headEqual {
			uasHits++
		}
		if headEqual && depEqual {
			lasHits++
		}
		if !headEqual || !depEqual {
			r.ArcDisagreements = append(r.ArcDisagreements, ArcDisagreement{
				Index: i, WantHead: w.Head, GotHead: g.Head, WantDep: w.Dep, GotDep: g.Dep,
			})
		}
	}
	n := float64(len(want))
	r.UAS = float64(uasHits) / n
	r.LAS = float64(lasHits) / n
	return r
}
