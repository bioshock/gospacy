// Package parserinternals ports the C++/Cython StateC and ArcEager TransitionSystem
// machinery from spacy.pipeline._parser_internals. Inference-only: oracle, gold
// costs, and beam search are out of scope.
package parserinternals

// Arc is a directed labelled arc between two tokens. label is the StringStore
// hash of the dep label (e.g. ss.Add("nsubj")). Mirrors ArcC from _state.pxd.
type Arc struct {
	Head  int
	Child int
	Label uint64
}

// Entity is a half-open [Start, End) entity span with a label hash. Mirrors
// the SpanC payload that backs Doc.ents in spaCy. Recorded by BiluoApply on
// LAST / UNIT.
type Entity struct {
	Start int
	End   int
	Label uint64
}

// State is the Go equivalent of StateC for a single Doc. All indices are
// 0-based token positions within the Doc. Greedy single-doc parsing only —
// the upstream `offset` field that supports multi-doc concatenation is not
// modelled.
type State struct {
	Length int
	stack  []int
	// rebuffer is the LIFO of unshifted tokens; B(i) consults this first
	// (in reverse order) before walking the linear buffer at bI.
	rebuffer []int
	bI       int
	// heads[i] is the head of token i (-1 = unattached). Mirrors StateC._heads.
	heads []int
	// leftArcs / rightArcs keyed by head — the L(head, idx) / R(head, idx)
	// nth-child accessors walk these in reverse order, matching nth_child
	// in _state.pxd.
	leftArcs  map[int][]Arc
	rightArcs map[int][]Arc
	// unshiftable[i]==true means Shift refuses to push i again.
	unshiftable []bool
	// sentStarts[i]==true means Break has marked i as a sentence start. The
	// initial token is implicitly a sentence start (init_state contract).
	sentStarts []bool

	// NER fields. Dormant when the State is driven by ArcEager (parser); used
	// only by BiluoApply / BiluoIsValid. Kept on the shared State per Phase 7
	// Decision 2 (Block D2): the field overlap with the parser is ~80%, and
	// adding a parallel NERState type would duplicate the buffer/feature
	// machinery without buying anything.
	//
	// entOpen is the start-token index of the currently open entity (-1 when
	// no entity is open). entOpenLabel is its StringStore hash. entIOB
	// records the per-token BILUO writeback (0 missing, 1 I-, 2 O, 3 B-) —
	// the same encoding as Token.EntIOB. entities accumulates closed spans
	// (UNIT or LAST emits).
	entOpen      int
	entOpenLabel uint64
	entIOB       []uint8
	entities     []Entity
}

// NewState builds an empty state for a Doc of n tokens. Token 0 is implicitly
// the start of the first sentence (matches upstream init_state).
func NewState(n int) *State {
	heads := make([]int, n)
	for i := range heads {
		heads[i] = -1
	}
	ss := make([]bool, n)
	if n > 0 {
		ss[0] = true
	}
	return &State{
		Length:       n,
		heads:        heads,
		leftArcs:     map[int][]Arc{},
		rightArcs:    map[int][]Arc{},
		unshiftable:  make([]bool, n),
		sentStarts:   ss,
		entOpen:      -1,
		entOpenLabel: 0,
		entIOB:       make([]uint8, n),
	}
}

// S returns the i-th token from the top of the stack, or -1.
func (s *State) S(i int) int {
	if i < 0 || i >= len(s.stack) {
		return -1
	}
	return s.stack[len(s.stack)-1-i]
}

// B returns the i-th token from the front of the buffer, or -1.
func (s *State) B(i int) int {
	if i < 0 {
		return -1
	}
	if i < len(s.rebuffer) {
		return s.rebuffer[len(s.rebuffer)-1-i]
	}
	bi := s.bI + (i - len(s.rebuffer))
	if bi >= s.Length {
		return -1
	}
	return bi
}

// H returns the head of child, or -1 if none.
func (s *State) H(child int) int {
	if child < 0 || child >= s.Length {
		return -1
	}
	return s.heads[child]
}

// L returns the idx-th left child of head (1-indexed; idx<1 returns -1).
// Matches nth_child semantics: walks the arc list in reverse insertion order.
func (s *State) L(head, idx int) int {
	if idx < 1 || head < 0 {
		return -1
	}
	arcs := s.leftArcs[head]
	count := 0
	for j := len(arcs) - 1; j >= 0; j-- {
		if arcs[j].Child < 0 {
			continue
		}
		count++
		if count == idx {
			return arcs[j].Child
		}
	}
	return -1
}

// R returns the idx-th right child of head (mirrors L on rightArcs).
func (s *State) R(head, idx int) int {
	if idx < 1 || head < 0 {
		return -1
	}
	arcs := s.rightArcs[head]
	count := 0
	for j := len(arcs) - 1; j >= 0; j-- {
		if arcs[j].Child < 0 {
			continue
		}
		count++
		if count == idx {
			return arcs[j].Child
		}
	}
	return -1
}

// StackDepth returns len(stack).
func (s *State) StackDepth() int { return len(s.stack) }

// BufferLength returns the number of buffer tokens still unprocessed (including
// rebuffered ones). Mirrors StateC.buffer_length.
func (s *State) BufferLength() int {
	return (s.Length - s.bI) + len(s.rebuffer)
}

// IsFinal returns true when there are no buffer tokens AND the stack is empty.
func (s *State) IsFinal() bool {
	return s.StackDepth() <= 0 && s.BufferLength() == 0
}

// Push moves the front buffer token onto the stack. If rebuffer is non-empty,
// pops from rebuffer first; otherwise advances bI.
func (s *State) Push() {
	var b0 int
	if n := len(s.rebuffer); n > 0 {
		b0 = s.rebuffer[n-1]
		s.rebuffer = s.rebuffer[:n-1]
	} else {
		b0 = s.bI
		s.bI++
	}
	s.stack = append(s.stack, b0)
}

// Pop removes the top of the stack.
func (s *State) Pop() {
	if n := len(s.stack); n > 0 {
		s.stack = s.stack[:n-1]
	}
}

// Unshift moves the top of the stack back to the front of the buffer, marking
// it as unshiftable so Shift cannot pick it up again on the same state.
func (s *State) Unshift() {
	n := len(s.stack)
	if n == 0 {
		return
	}
	s0 := s.stack[n-1]
	s.unshiftable[s0] = true
	s.rebuffer = append(s.rebuffer, s0)
	s.stack = s.stack[:n-1]
}

// AddArc records head→child with label. If child already has a head, the
// previous arc is removed first (mirrors StateC.add_arc).
func (s *State) AddArc(head, child int, label uint64) {
	if s.HasHead(child) {
		s.delArc(s.heads[child], child)
	}
	arc := Arc{Head: head, Child: child, Label: label}
	if head > child {
		s.leftArcs[head] = append(s.leftArcs[head], arc)
	} else {
		s.rightArcs[head] = append(s.rightArcs[head], arc)
	}
	s.heads[child] = head
}

func (s *State) delArc(head, child int) {
	bucket := &s.rightArcs
	if head > child {
		bucket = &s.leftArcs
	}
	arcs := (*bucket)[head]
	for i := len(arcs) - 1; i >= 0; i-- {
		if arcs[i].Head == head && arcs[i].Child == child {
			if i == len(arcs)-1 {
				(*bucket)[head] = arcs[:i]
			} else {
				arcs[i].Head = -1
				arcs[i].Child = -1
				arcs[i].Label = 0
				(*bucket)[head] = arcs
			}
			return
		}
	}
}

// HasHead reports whether child has a recorded head.
func (s *State) HasHead(child int) bool {
	return child >= 0 && child < s.Length && s.heads[child] >= 0
}

// IsUnshiftable returns 1 if item has been marked unshiftable, else 0 (int
// return matches the upstream is_unshiftable signature used by Shift.is_valid).
func (s *State) IsUnshiftable(item int) int {
	if item < 0 || item >= len(s.unshiftable) {
		return 0
	}
	if s.unshiftable[item] {
		return 1
	}
	return 0
}

// SetReshiftable clears the unshiftable bit for item.
func (s *State) SetReshiftable(item int) {
	if item >= 0 && item < len(s.unshiftable) {
		s.unshiftable[item] = false
	}
}

// IsSentStart returns 1 if word is a sentence start, else 0. Mirrors
// StateC.is_sent_start. Token-level overrides (sent_start == -1 set by the
// tokenizer) are not modelled — gospacy's tokenizer leaves SentStart == 0.
func (s *State) IsSentStart(word int) int {
	if word < 0 || word >= s.Length {
		return 0
	}
	if s.sentStarts[word] {
		return 1
	}
	return 0
}

// SetSentStart marks word as a sentence start when v >= 1.
func (s *State) SetSentStart(word, v int) {
	if word < 0 || word >= s.Length || v < 1 {
		return
	}
	s.sentStarts[word] = true
}

// CannotSentStart returns 1 if word is explicitly forbidden from being a
// sentence start (TokenC.sent_start == -1). gospacy's tokenizer never sets
// that flag → always returns 0 here, but the method is part of the Break.cost
// signature and Reduce.is_valid uses it for the single-stack edge case.
func (s *State) CannotSentStart(word int) int { return 0 }

// ForceFinal drains the buffer and the stack — used when arg_max_if_valid
// returns -1 (no valid move).
func (s *State) ForceFinal() {
	s.stack = s.stack[:0]
	s.bI = s.Length
	s.rebuffer = s.rebuffer[:0]
}

// SetContextTokens8 fills the 8-feature context array used by the parser's
// PrecomputableAffine lower layer. -1 means "missing"; the scorer routes
// missing features through the pad row.
func (s *State) SetContextTokens8(ids *[8]int32) {
	ids[0] = int32(s.B(0))
	ids[1] = int32(s.B(1))
	ids[2] = int32(s.S(0))
	ids[3] = int32(s.S(1))
	ids[4] = int32(s.S(2))
	ids[5] = int32(s.L(s.B(0), 1))
	ids[6] = int32(s.L(s.S(0), 1))
	ids[7] = int32(s.R(s.S(0), 1))
}

// SetContextTokens3NER fills the 3-feature context array used by the NER
// PrecomputableAffine lower (en_core_web_sm/md/lg all use nF=3 for NER).
// Mirrors the n==3 branch of StateC.set_context_tokens in
// _parser_internals/_state.pxd:
//
//	ids[0] = B(0)               or -1 if no buffer token
//	ids[1] = E(0)               (entity start) or -1 if no entity open
//	ids[2] = ids[0] - 1         or -1 if either prior is -1
//
// -1 routes through the lower's pad row, the same convention as the
// 8-feature parser template.
func (s *State) SetContextTokens3NER(ids *[3]int32) {
	b0 := s.B(0)
	if b0 < 0 {
		ids[0] = -1
	} else {
		ids[0] = int32(b0)
	}
	if s.EntityIsOpen() {
		ids[1] = int32(s.EntStart())
	} else {
		ids[1] = -1
	}
	if ids[0] == -1 || ids[1] == -1 {
		ids[2] = -1
	} else {
		ids[2] = ids[0] - 1
	}
}

// Arcs returns every recorded arc, in no particular order. Used by Parser.Apply
// to write Token.Head / Token.Dep after parsing.
func (s *State) Arcs() []Arc {
	out := make([]Arc, 0)
	for _, arcs := range s.leftArcs {
		for _, a := range arcs {
			if a.Head != -1 && a.Child != -1 {
				out = append(out, a)
			}
		}
	}
	for _, arcs := range s.rightArcs {
		for _, a := range arcs {
			if a.Head != -1 && a.Child != -1 {
				out = append(out, a)
			}
		}
	}
	return out
}

// EntStart returns the start-token index of the currently open entity, or -1
// when no entity is open. Mirrors StateC.E(0).
func (s *State) EntStart() int { return s.entOpen }

// EntLabel returns the StringStore hash of the currently open entity's label,
// or 0 when no entity is open.
func (s *State) EntLabel() uint64 { return s.entOpenLabel }

// EntityIsOpen reports whether s currently has an open entity. Mirrors
// StateC.entity_is_open.
func (s *State) EntityIsOpen() bool { return s.entOpen >= 0 }

// OpenEntity starts a new entity at start with label hash. Callers should
// CloseEntity or RecordEntity first; this is permissive to avoid a panic on
// adversarial inputs.
func (s *State) OpenEntity(start int, labelHash uint64) {
	s.entOpen = start
	s.entOpenLabel = labelHash
}

// CloseEntity clears the open-entity register without recording a span. Used
// by Apply when a malformed sequence forces a discard.
func (s *State) CloseEntity() {
	s.entOpen = -1
	s.entOpenLabel = 0
}

// RecordEntity appends a finalized entity span [start, end) with label hash
// to s.entities and clears the open register.
func (s *State) RecordEntity(start, end int, labelHash uint64) {
	s.entities = append(s.entities, Entity{Start: start, End: end, Label: labelHash})
	s.CloseEntity()
}

// SetEntIOB writes the BILUO code for token i. Out-of-range i is silently
// ignored to keep callers concise (mirrors the bounds-check pattern in
// SetSentStart / SetReshiftable).
func (s *State) SetEntIOB(i int, iob uint8) {
	if i >= 0 && i < len(s.entIOB) {
		s.entIOB[i] = iob
	}
}

// EntIOB returns the BILUO code for token i (0 missing, 1 I-, 2 O, 3 B-). The
// encoding matches Token.EntIOB so NER.Apply can write it directly.
func (s *State) EntIOB(i int) uint8 {
	if i < 0 || i >= len(s.entIOB) {
		return 0
	}
	return s.entIOB[i]
}

// Entities returns a copy of the recorded entity spans. Used by NER.Apply to
// write Token.EntType after the BILUO sweep finishes.
func (s *State) Entities() []Entity {
	out := make([]Entity, len(s.entities))
	copy(out, s.entities)
	return out
}
