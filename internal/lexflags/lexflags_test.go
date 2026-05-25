package lexflags

import "testing"

func TestIsAlpha(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"", false},
		{"hello", true},
		{"HELLO", true},
		{"café", true}, // accented letter
		{"hello1", false},
		{"hello!", false},
		{"123", false},
		{"  ", false},
	}
	for _, c := range cases {
		if got := IsAlpha(c.in); got != c.want {
			t.Errorf("IsAlpha(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestIsPunct(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"", false},
		{".", true},
		{"!", true},
		{",.", true},
		{"--", true},
		{"a", false},
		{"a.", false},
		{"...", true},
	}
	for _, c := range cases {
		if got := IsPunct(c.in); got != c.want {
			t.Errorf("IsPunct(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestIsDigit(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"", false},
		{"0", true},
		{"123", true},
		{"1.5", false}, // period isn't a digit
		{"abc", false},
		{"1a", false},
	}
	for _, c := range cases {
		if got := IsDigit(c.in); got != c.want {
			t.Errorf("IsDigit(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestLikeNum_MatchesPythonSpaCy covers every branch of
// spacy/lang/en/lex_attrs.py:like_num. Each case has been
// verified against the Python implementation.
func TestLikeNum(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		// Pure digits.
		{"42", true},
		{"0", true},
		// Signed digits (sign is stripped).
		{"+42", true},
		{"-42", true},
		{"±42", true},
		{"~42", true},
		// Digits with commas / periods (stripped).
		{"1,000", true},
		{"3.14", true},
		{"1,000,000", true},
		// Fractions.
		{"3/4", true},
		{"1/2", true},
		{"3/", false},   // denom missing
		{"3//4", false}, // two slashes
		// Cardinal words (case-insensitive).
		{"one", true},
		{"FORTY", true},
		{"Billion", true},
		// Ordinal words.
		{"first", true},
		{"thirteenth", true},
		// Suffix form.
		{"21st", true},
		{"32nd", true},
		{"103rd", true},
		{"4th", true},
		{"1st", true},
		// Non-numerics.
		{"", false},
		{"hello", false},
		{"abc", false},
		{"nstrd", false}, // not a digit prefix
	}
	for _, c := range cases {
		if got := LikeNum(c.in); got != c.want {
			t.Errorf("LikeNum(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}
