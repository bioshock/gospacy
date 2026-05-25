package parserinternals

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestArcEager_ShiftValidity walks the initial state and confirms the only
// stack-dependent moves (Reduce/Left/Right) are invalid; Shift is valid
// (stack empty rule); Break is valid because B(1)==B(0)+1 and B(1) is not
// already a sentence start. Mirrors the Cython is_valid bodies exactly.
func TestArcEager_ShiftValidity(t *testing.T) {
	s := NewState(3)
	require.Equal(t, 1, IsValid(s, ActionShift, "", 0))
	require.Equal(t, 0, IsValid(s, ActionReduce, "", 0))
	require.Equal(t, 0, IsValid(s, ActionLeft, "nsubj", 1))
	require.Equal(t, 0, IsValid(s, ActionRight, "dobj", 2))
	require.Equal(t, 1, IsValid(s, ActionBreak, "ROOT", 0))
}

// TestArcEager_LeftArcMakesParentChildAndPops mirrors a tiny "she sees him"
// parse: SHIFT (she), LEFT-nsubj (sees ← she), SHIFT (sees), RIGHT-dobj (sees → him).
func TestArcEager_LeftArcMakesParentChildAndPops(t *testing.T) {
	s := NewState(3) // tokens: 0=she, 1=sees, 2=him
	Apply(s, ActionShift, "", 0)
	require.Equal(t, 0, s.S(0))
	Apply(s, ActionLeft, "nsubj", 100)
	require.Equal(t, 1, s.H(0))             // she ← sees
	require.Equal(t, uint64(100), s.heads0Label())
	require.Equal(t, 0, s.StackDepth())     // S(0) popped
	Apply(s, ActionShift, "", 0)            // shift "sees"
	Apply(s, ActionRight, "dobj", 200)
	require.Equal(t, 1, s.H(2))             // him ← sees (right arc)
	// After right-arc we push B(0)=him onto the stack.
	require.Equal(t, 2, s.S(0))
}

// TestArcEager_ReduceUnshiftsIfNoHead — Reduce on a headless S(0) places it
// back on the buffer (unshift), not pop.
func TestArcEager_ReduceUnshiftsIfNoHead(t *testing.T) {
	s := NewState(3)
	Apply(s, ActionShift, "", 0) // stack=[0]
	Apply(s, ActionShift, "", 0) // stack=[0,1]
	require.False(t, s.HasHead(1))
	Apply(s, ActionReduce, "", 0)
	require.Equal(t, 1, s.IsUnshiftable(1))
	// Unshifted token returns to B(0).
	require.Equal(t, 1, s.B(0))
}

// TestArcEager_BreakSetsSentStart confirms Break marks B(1) as a sentence
// boundary. Validity needs buffer_length() >= 2 (so we use a 3-token state
// before any shift to keep both B(0) and B(1) live).
func TestArcEager_BreakSetsSentStart(t *testing.T) {
	s := NewState(3) // tokens 0,1,2 → BufferLength=3, B(0)=0, B(1)=1
	require.Equal(t, 1, IsValid(s, ActionBreak, "ROOT", 0))
	Apply(s, ActionBreak, "ROOT", 0)
	require.Equal(t, 1, s.IsSentStart(1))
	// After Break, repeating it on the same sentence boundary becomes invalid
	// because B(1) is now marked sent_start.
	require.Equal(t, 0, IsValid(s, ActionBreak, "ROOT", 0))
	// Buffer < 2 also fails validity.
	s2 := NewState(2)
	Apply(s2, ActionShift, "", 0) // BufferLength=1
	require.Equal(t, 0, IsValid(s2, ActionBreak, "ROOT", 0))
}

// TestArcEager_SetValid walks a tiny transition table and confirms SetValid
// stamps the SUBTOK label restriction (LEFT/RIGHT with label=="subtok" demand
// S(0) == B(0)-1; otherwise return move-level validity).
func TestArcEager_SetValid(t *testing.T) {
	ts := &TransitionSystem{
		Transitions: []Transition{
			{ClassIndex: 0, Move: ActionShift, Label: ""},
			{ClassIndex: 1, Move: ActionReduce, Label: ""},
			{ClassIndex: 2, Move: ActionLeft, Label: "nsubj"},
			{ClassIndex: 3, Move: ActionLeft, Label: "subtok"},
		},
		NMoves: 4,
	}
	s := NewState(4)
	Apply(s, ActionShift, "", 0) // stack=[0]; B(0)=1
	out := make([]int32, 4)
	SetValid(out, s, ts)
	// SHIFT valid (buffer >= 2, stack non-empty).
	require.Equal(t, int32(1), out[0])
	// REDUCE invalid when stack token has no head AND stack_depth>1 path doesn't apply.
	// (See Reduce.is_valid: with stack non-empty + buffer non-empty + depth==1 the
	// cannot_sent_start check decides; on a fresh State CannotSentStart is 0 so
	// Reduce is VALID here.)
	require.Equal(t, int32(1), out[1])
	// LEFT-nsubj is valid (stack non-empty, B(0) not sent_start at idx 1).
	require.Equal(t, int32(1), out[2])
	// LEFT-subtok demands S(0) == B(0)-1 == 0; that holds, so valid.
	require.Equal(t, int32(1), out[3])

	// Build a state where S(0) is NOT adjacent to B(0): shift, shift, left-arc.
	// shift: stack=[0], B(0)=1
	// shift: stack=[0,1], B(0)=2
	// left-arc(nsubj): AddArc(B(0)=2, S(0)=1), pop S(0) → stack=[0], B(0)=2.
	// Now S(0)=0, B(0)=2, B(0)-1=1 ≠ 0 → SUBTOK gate should invalidate.
	gap := NewState(5)
	Apply(gap, ActionShift, "", 0)
	Apply(gap, ActionShift, "", 0)
	Apply(gap, ActionLeft, "nsubj", 42)
	require.Equal(t, 0, gap.S(0))
	require.Equal(t, 2, gap.B(0))
	out2 := make([]int32, 4)
	SetValid(out2, gap, ts)
	require.Equal(t, int32(1), out2[2]) // plain LEFT-nsubj still valid
	require.Equal(t, int32(0), out2[3]) // LEFT-subtok now invalid (S(0)=0, B(0)-1=1)
}

// heads0Label peeks at the label of the arc whose child is 0. Used by
// TestArcEager_LeftArcMakesParentChildAndPops to read back what Apply wrote.
func (s *State) heads0Label() uint64 {
	for _, a := range s.Arcs() {
		if a.Child == 0 {
			return a.Label
		}
	}
	return 0
}
