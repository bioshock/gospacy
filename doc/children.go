package doc

import "sort"

// ensureChildIdx lazily builds the CSR children index on d from
// Tokens[].Head. Idempotent — checking childStart != nil avoids rebuild on
// every Children/Subtree call. Single-goroutine by gospacy's contract
// (one Doc per goroutine), so no synchronisation needed.
//
// Build cost: O(N). After the first call every ChildrenOf is O(k) where
// k is the number of children of the queried token; every SubtreeOf is
// O(subtree-size). Replaces the pre-cache O(N) per ChildrenOf / O(N×S)
// per SubtreeOf.
func (d *Doc) ensureChildIdx() {
	if d.childStart != nil {
		return
	}
	n := len(d.Tokens)
	if n == 0 {
		d.childStart = []int32{0}
		d.childIdx = nil
		return
	}
	counts := make([]int32, n)
	for i := 0; i < n; i++ {
		h := d.Tokens[i].Head
		if h == i {
			continue // root self-points; not its own child
		}
		if h >= 0 && h < n {
			counts[h]++
		}
	}
	d.childStart = make([]int32, n+1)
	for i := 0; i < n; i++ {
		d.childStart[i+1] = d.childStart[i] + counts[i]
	}
	d.childIdx = make([]int32, d.childStart[n])
	pos := make([]int32, n)
	copy(pos, d.childStart[:n])
	for i := 0; i < n; i++ {
		h := d.Tokens[i].Head
		if h == i {
			continue
		}
		if h >= 0 && h < n {
			d.childIdx[pos[h]] = int32(i)
			pos[h]++
		}
	}
	// children are inserted in ascending token-index order because i runs
	// ascending, so no separate sort is needed.
}

// Children returns the indices of tokens whose Head points at t's own index
// in the parent Doc, excluding t itself (the parse root has Head == its own
// index, so we filter that out). Results are in ascending token order.
//
// Caller passes in the owning Doc and the token's own index because Token is
// a value type — it doesn't know its own position. Use d.Tokens[i].Children(d, i)
// or the convenience function ChildrenOf(d, i) below.
func ChildrenOf(d *Doc, self int) []int {
	d.ensureChildIdx()
	if self < 0 || self >= len(d.Tokens) {
		return nil
	}
	s, e := d.childStart[self], d.childStart[self+1]
	if s == e {
		return nil
	}
	out := make([]int, e-s)
	for k, v := range d.childIdx[s:e] {
		out[k] = int(v)
	}
	return out
}

// SubtreeOf returns the indices of t and every descendant in t's subtree
// (transitive children), sorted ascending. Equivalent to spaCy's
// `[w.i for w in token.subtree]`. Caller passes in the owning Doc and
// self-index because Token is a value type.
func SubtreeOf(d *Doc, self int) []int {
	d.ensureChildIdx()
	if self < 0 || self >= len(d.Tokens) {
		return nil
	}
	seen := map[int]struct{}{self: {}}
	queue := []int{self}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		s, e := d.childStart[cur], d.childStart[cur+1]
		for _, ch := range d.childIdx[s:e] {
			ci := int(ch)
			if _, ok := seen[ci]; ok {
				continue
			}
			seen[ci] = struct{}{}
			queue = append(queue, ci)
		}
	}
	out := make([]int, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}
