package parserinternals

import (
	"strings"

	"github.com/bioshock/gospacy/v3/doc"
)

// Delimiter mirrors nonproj.DELIMITER (nonproj.pyx:21) — the marker spaCy
// uses when decorating a label whose arc was lifted to the grandfather.
const Delimiter = "||"

// Deprojectivize walks the Doc and re-attaches every token whose Dep label
// contains "||" to the first BFS-descendant of the current head that bears
// the head_label half of the decoration. Mirrors cpdef deprojectivize
// (nonproj.pyx:176).
//
// Pure projective bundles (like en_core_web_sm's 8-sentence fixtures) never
// hit the decoration path; this function is a guarded no-op there.
func Deprojectivize(d *doc.Doc) {
	if d == nil || d.NumTokens() == 0 {
		return
	}
	ss := d.Vocab.StringStore()
	for i := range d.Tokens {
		label, ok := ss.Lookup(d.Tokens[i].Dep)
		if !ok || !strings.Contains(label, Delimiter) {
			continue
		}
		parts := strings.SplitN(label, Delimiter, 2)
		newLabel, headLabel := parts[0], parts[1]
		newHead := findNewHead(d, i, headLabel)
		if newHead >= 0 {
			d.Tokens[i].Head = newHead
		}
		d.Tokens[i].Dep = ss.Add(newLabel)
	}
}

// findNewHead BFS-searches from the current head of tok i for the first
// descendant (skipping i itself) whose Dep is headLabel. Returns the
// descendant's index or the current head's index when nothing matches.
// Mirrors _find_new_head (nonproj.pyx:241).
func findNewHead(d *doc.Doc, tokIdx int, headLabel string) int {
	head := d.Tokens[tokIdx].Head
	if head < 0 || head >= d.NumTokens() {
		return head
	}
	ss := d.Vocab.StringStore()
	target := headLabel
	// children[i] is the list of token indices whose Head == i.
	children := make([][]int, d.NumTokens())
	for j := range d.Tokens {
		h := d.Tokens[j].Head
		if h >= 0 && h < d.NumTokens() && h != j {
			children[h] = append(children[h], j)
		}
	}
	queue := append([]int(nil), head)
	for len(queue) > 0 {
		next := queue[:0]
		for _, qi := range queue {
			for _, child := range children[qi] {
				if child == tokIdx {
					continue
				}
				lbl, _ := ss.Lookup(d.Tokens[child].Dep)
				if lbl == target {
					return child
				}
				next = append(next, child)
			}
		}
		queue = next
	}
	return head
}
