package parserinternals

// SubtokLabel is the label string spaCy uses for token-merge LEFT/RIGHT arcs.
// Both LEFT and RIGHT validity require S(0) == B(0)-1 when the label is
// "subtok" (arc_eager.pyx:431 and 484). en_core_web_sm's parser/moves has no
// entries with label="subtok" (learn_tokens=false), so the gate is a no-op
// here, but we honour it for correctness.
const SubtokLabel = "subtok"

// IsValid mirrors the per-action is_valid methods in arc_eager.pyx. label is
// the move's label string (used only for the SUBTOK gate). labelHash is
// unused at validity-check time — it's an Apply parameter, kept in the
// signature so callers don't have to switch on Move.
func IsValid(s *State, move int, label string, _ uint64) int {
	switch move {
	case ActionShift:
		// Shift.is_valid: stack empty OR (buffer >= 2 AND B(0) not sent_start AND
		// B(0) not unshiftable).
		if s.StackDepth() == 0 {
			if s.BufferLength() == 0 {
				return 0
			}
			return 1
		}
		if s.BufferLength() < 2 {
			return 0
		}
		if s.IsSentStart(s.B(0)) == 1 {
			return 0
		}
		if s.IsUnshiftable(s.B(0)) == 1 {
			return 0
		}
		return 1

	case ActionReduce:
		// Reduce.is_valid: stack non-empty AND (buffer empty OR stack-depth>1 OR
		// !cannot_sent_start(l_edge(B(0)))). l_edge(x) == x in StateC, so the
		// check is cannot_sent_start(B(0)).
		if s.StackDepth() == 0 {
			return 0
		}
		if s.BufferLength() == 0 {
			return 1
		}
		if s.StackDepth() == 1 && s.CannotSentStart(s.B(0)) == 1 {
			return 0
		}
		return 1

	case ActionLeft:
		// LeftArc.is_valid: stack non-empty, buffer non-empty, B(0) not
		// sent_start. SUBTOK label demands S(0) == B(0)-1.
		if s.StackDepth() == 0 || s.BufferLength() == 0 {
			return 0
		}
		if s.IsSentStart(s.B(0)) == 1 {
			return 0
		}
		if label == SubtokLabel && s.S(0) != s.B(0)-1 {
			return 0
		}
		return 1

	case ActionRight:
		// Same shape as LeftArc.is_valid.
		if s.StackDepth() == 0 || s.BufferLength() == 0 {
			return 0
		}
		if s.IsSentStart(s.B(0)) == 1 {
			return 0
		}
		if label == SubtokLabel && s.S(0) != s.B(0)-1 {
			return 0
		}
		return 1

	case ActionBreak:
		// Break.is_valid: buffer >= 2 AND B(1) == B(0)+1 AND !sent_start(B(1))
		// AND !cannot_sent_start(B(1)).
		if s.BufferLength() < 2 {
			return 0
		}
		if s.B(1) != s.B(0)+1 {
			return 0
		}
		if s.IsSentStart(s.B(1)) == 1 {
			return 0
		}
		if s.CannotSentStart(s.B(1)) == 1 {
			return 0
		}
		return 1
	}
	return 0
}

// Apply mutates s according to the (move, label) action. labelHash is the
// StringStore hash of label (resolved once by the caller, e.g. Parser.Apply).
func Apply(s *State, move int, label string, labelHash uint64) {
	switch move {
	case ActionShift:
		s.Push()

	case ActionReduce:
		s0 := s.S(0)
		if s.HasHead(s0) || s.StackDepth() == 1 {
			s.Pop()
		} else {
			s.Unshift()
		}

	case ActionLeft:
		s.AddArc(s.B(0), s.S(0), labelHash)
		s.SetReshiftable(s.B(0))
		s.Pop()

	case ActionRight:
		s.AddArc(s.S(0), s.B(0), labelHash)
		s.Push()

	case ActionBreak:
		s.SetSentStart(s.B(1), 1)
	}
}

// SetValid stamps out[i] = IsValid(s, ts.Transitions[i].Move, ts.Transitions[i].Label).
// Mirrors ArcEager.set_valid (arc_eager.pyx:788). Caller pre-sizes out to ts.NMoves.
func SetValid(out []int32, s *State, ts *TransitionSystem) {
	// Per the upstream optimisation: compute move-level validity once and
	// short-circuit; only SUBTOK-labelled transitions need a per-class re-check.
	moveValid := [5]int32{
		int32(IsValid(s, ActionShift, "", 0)),
		int32(IsValid(s, ActionReduce, "", 0)),
		int32(IsValid(s, ActionLeft, "", 0)),
		int32(IsValid(s, ActionRight, "", 0)),
		int32(IsValid(s, ActionBreak, "", 0)),
	}
	for i, tr := range ts.Transitions {
		if tr.Label == SubtokLabel {
			out[i] = int32(IsValid(s, tr.Move, tr.Label, 0))
		} else {
			out[i] = moveValid[tr.Move]
		}
	}
}
