package diff

import "fmt"

// DisagreeKind classifies a token-level disagreement.
type DisagreeKind int

const (
	DisagreeLength DisagreeKind = iota + 1 // sequences have different lengths
	DisagreeOrth                           // token text differs
	DisagreeIdx                            // token byte offset differs
	DisagreeWS                             // trailing-whitespace flag differs
)

// String returns a human-readable name for the DisagreeKind constant.
func (k DisagreeKind) String() string {
	switch k {
	case DisagreeLength:
		return "length"
	case DisagreeOrth:
		return "orth"
	case DisagreeIdx:
		return "idx"
	case DisagreeWS:
		return "ws"
	default:
		return fmt.Sprintf("unknown(%d)", int(k))
	}
}

// Disagreement records one mismatch between expected and actual.
type Disagreement struct {
	Kind  DisagreeKind
	Index int    // token index where disagreement occurred (or -1 for length)
	Want  string // human-readable expected value
	Got   string // human-readable actual value
}

// Report is the comparator output. Empty Disagreements means equal.
type Report struct {
	Disagreements []Disagreement
}

// Equal reports whether the comparison found no disagreements.
func (r *Report) Equal() bool { return len(r.Disagreements) == 0 }

// CompareTokens diffs two token sequences and returns a Report.
// On length mismatch it stops at the length disagreement and does not compare
// individual tokens (avoid noisy cascades).
func CompareTokens(want, got []Token) Report {
	var r Report
	if len(want) != len(got) {
		r.Disagreements = append(r.Disagreements, Disagreement{
			Kind:  DisagreeLength,
			Index: -1,
			Want:  fmt.Sprintf("%d tokens", len(want)),
			Got:   fmt.Sprintf("%d tokens", len(got)),
		})
		return r
	}
	for i := range want {
		w, g := want[i], got[i]
		if w.Orth != g.Orth {
			r.Disagreements = append(r.Disagreements, Disagreement{
				Kind: DisagreeOrth, Index: i, Want: w.Orth, Got: g.Orth,
			})
			continue // a wrong orth probably invalidates subsequent comparisons
		}
		if w.Idx != g.Idx {
			r.Disagreements = append(r.Disagreements, Disagreement{
				Kind: DisagreeIdx, Index: i,
				Want: fmt.Sprintf("%d", w.Idx), Got: fmt.Sprintf("%d", g.Idx),
			})
		}
		if w.WS != g.WS {
			r.Disagreements = append(r.Disagreements, Disagreement{
				Kind: DisagreeWS, Index: i,
				Want: fmt.Sprintf("%t", w.WS), Got: fmt.Sprintf("%t", g.WS),
			})
		}
	}
	return r
}
