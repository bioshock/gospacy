package diff

// Class is a coarse bucket for a token-level Report.
type Class int

const (
	ClassEqual          Class = iota // no disagreements
	ClassLengthMismatch              // sequences differ in length — usually means tokenizer split differently
	ClassWhitespaceOnly              // only DisagreeWS disagreements — usually benign
	ClassOffsetOnly                  // only DisagreeIdx disagreements — usually upstream whitespace handling
	ClassRealDivergence              // includes DisagreeOrth or mixed kinds — needs investigation
)

// String returns a snake_case label for the Class constant, suitable for test
// failure messages and log output.
func (c Class) String() string {
	switch c {
	case ClassEqual:
		return "equal"
	case ClassLengthMismatch:
		return "length_mismatch"
	case ClassWhitespaceOnly:
		return "whitespace_only"
	case ClassOffsetOnly:
		return "offset_only"
	case ClassRealDivergence:
		return "real_divergence"
	default:
		return "unknown"
	}
}

// Classification summarises a Report into a coarse bucket plus per-kind counts.
type Classification struct {
	Primary Class
	Counts  map[DisagreeKind]int
}

// Classify maps a Report into a coarse Class bucket and per-kind disagreement
// counts. It prefers the most informative bucket: length mismatches and orth
// differences are escalated above whitespace/offset-only divergences.
func Classify(r Report) Classification {
	c := Classification{Counts: map[DisagreeKind]int{}}
	if len(r.Disagreements) == 0 {
		c.Primary = ClassEqual
		return c
	}
	for _, d := range r.Disagreements {
		c.Counts[d.Kind]++
	}
	// Length mismatch is its own bucket (we short-circuit in CompareTokens, so it
	// will be the only disagreement when present).
	if c.Counts[DisagreeLength] > 0 {
		c.Primary = ClassLengthMismatch
		return c
	}
	hasOrth := c.Counts[DisagreeOrth] > 0
	hasIdx := c.Counts[DisagreeIdx] > 0
	hasWS := c.Counts[DisagreeWS] > 0
	switch {
	case hasOrth:
		c.Primary = ClassRealDivergence
	case hasIdx && !hasWS:
		c.Primary = ClassOffsetOnly
	case hasWS && !hasIdx:
		c.Primary = ClassWhitespaceOnly
	default:
		// any other mix
		c.Primary = ClassRealDivergence
	}
	return c
}
