package diff

import (
	"fmt"
	"math"
)

// Tolerance specifies how close two floats must be to count as equal.
// At each index, equality holds if either:
//   - |want - got| <= AbsMax (absolute tolerance, used near zero), OR
//   - |want - got| / |want| <= RelMax (relative tolerance, used far from zero)
// Set RelMax = 0 to disable relative checking.
type Tolerance struct {
	AbsMax float32
	RelMax float32
}

// NumericReport summarises a float-array comparison.
type NumericReport struct {
	LengthMismatch    string  // non-empty if lengths differ; describes "want N got M"
	HasNaN            bool    // any NaN in either array (always a failure)
	FirstDisagreeIdx  int     // -1 if equal, else index of first failure
	MaxAbsDiff        float32 // max |want[i]-got[i]| over all i
	MaxRelDiff        float32 // max |want[i]-got[i]| / |want[i]| over indices where want[i] != 0
	MeanAbsDiff       float32 // mean |want[i]-got[i]| over all i
}

// Equal reports whether the comparison found no length mismatch, no NaNs, and
// all elements within the specified tolerance.
func (r *NumericReport) Equal() bool {
	return r.LengthMismatch == "" && !r.HasNaN && r.FirstDisagreeIdx == -1
}

// CompareFloats diffs two float32 slices element-by-element within tol and
// returns a NumericReport with aggregate statistics. FirstDisagreeIdx is -1 if
// all elements pass; otherwise it holds the index of the first failing element.
func CompareFloats(want, got []float32, tol Tolerance) NumericReport {
	r := NumericReport{FirstDisagreeIdx: -1}
	if len(want) != len(got) {
		r.LengthMismatch = fmt.Sprintf("want %d got %d", len(want), len(got))
		return r
	}
	if len(want) == 0 {
		return r
	}
	var sumAbs float32
	for i := range want {
		w, g := want[i], got[i]
		if math32IsNaN(w) || math32IsNaN(g) {
			r.HasNaN = true
			if r.FirstDisagreeIdx == -1 {
				r.FirstDisagreeIdx = i
			}
			continue
		}
		abs := absF32(w - g)
		sumAbs += abs
		if abs > r.MaxAbsDiff {
			r.MaxAbsDiff = abs
		}
		var rel float32
		if w != 0 {
			rel = abs / absF32(w)
			if rel > r.MaxRelDiff {
				r.MaxRelDiff = rel
			}
		}
		// Pass if within abs tolerance OR (when want!=0) within rel tolerance.
		withinAbs := abs <= tol.AbsMax
		withinRel := w != 0 && tol.RelMax > 0 && rel <= tol.RelMax
		if !withinAbs && !withinRel {
			if r.FirstDisagreeIdx == -1 {
				r.FirstDisagreeIdx = i
			}
		}
	}
	r.MeanAbsDiff = sumAbs / float32(len(want))
	return r
}

// AssertFloats is a test helper that fails t if want and got differ by more than
// absTol (absolute tolerance). Pass absTol=0 to require exact equality.
// label is used in the failure message for context.
func AssertFloats(t interface {
	Helper()
	Fatalf(string, ...any)
}, want, got []float32, absTol float32, label string) {
	t.Helper()
	tol := Tolerance{AbsMax: absTol, RelMax: 0}
	rep := CompareFloats(want, got, tol)
	if !rep.Equal() {
		t.Fatalf("%s: float mismatch — len(want)=%d len(got)=%d firstDisagreeIdx=%d maxAbsDiff=%g",
			label, len(want), len(got), rep.FirstDisagreeIdx, rep.MaxAbsDiff)
	}
}

func math32IsNaN(x float32) bool { return math.IsNaN(float64(x)) }

func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
