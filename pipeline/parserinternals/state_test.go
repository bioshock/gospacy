package parserinternals

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestState_InitialIsBuffered(t *testing.T) {
	s := NewState(5)
	require.Equal(t, 5, s.Length)
	require.Equal(t, 0, s.StackDepth())
	require.Equal(t, 5, s.BufferLength())
	require.False(t, s.IsFinal())
	require.Equal(t, 0, s.B(0))
	require.Equal(t, 1, s.B(1))
	require.Equal(t, -1, s.S(0))
}

func TestState_PushPopAndArcs(t *testing.T) {
	s := NewState(4)
	s.Push() // stack=[0], buffer cursor=1
	require.Equal(t, 0, s.S(0))
	require.Equal(t, 1, s.B(0))
	s.Push() // stack=[0,1], buffer cursor=2
	s.AddArc(0, 1, 42)
	require.True(t, s.HasHead(1))
	require.Equal(t, 0, s.H(1))
	// First (and only) right-child of 0 is 1.
	require.Equal(t, 1, s.R(0, 1))
	s.Pop()
	require.Equal(t, 0, s.S(0))
	require.Equal(t, 1, s.StackDepth())
}

func TestState_UnshiftMarksUnshiftable(t *testing.T) {
	s := NewState(3)
	s.Push()
	require.Equal(t, 0, s.S(0))
	s.Unshift()
	require.Equal(t, 1, s.IsUnshiftable(0))
	require.Equal(t, 0, s.StackDepth())
	// After unshift, B(0) returns the unshifted token, not advancing.
	require.Equal(t, 0, s.B(0))
}

func TestState_SetContextTokens8(t *testing.T) {
	s := NewState(6)
	s.Push() // stack=[0]
	s.Push() // stack=[0,1]
	s.Push() // stack=[0,1,2], buf=[3,4,5]
	var ids [8]int32
	s.SetContextTokens8(&ids)
	require.Equal(t, int32(3), ids[0]) // B(0)
	require.Equal(t, int32(4), ids[1]) // B(1)
	require.Equal(t, int32(2), ids[2]) // S(0)
	require.Equal(t, int32(1), ids[3]) // S(1)
	require.Equal(t, int32(0), ids[4]) // S(2)
	// No arcs yet → L/R indices are -1.
	require.Equal(t, int32(-1), ids[5])
	require.Equal(t, int32(-1), ids[6])
	require.Equal(t, int32(-1), ids[7])
}

func TestState_IsFinal(t *testing.T) {
	s := NewState(2)
	s.Push()
	s.Push()
	// Buffer drained, stack still has 2 → not final.
	require.False(t, s.IsFinal())
	s.Pop()
	s.Pop()
	require.True(t, s.IsFinal())
}

func TestState_NER_OpenEntityRegister(t *testing.T) {
	s := NewState(5)
	require.False(t, s.EntityIsOpen(), "no entity open initially")
	require.Equal(t, -1, s.EntStart())
	require.Equal(t, uint64(0), s.EntLabel())

	s.OpenEntity(2, 42)
	require.True(t, s.EntityIsOpen())
	require.Equal(t, 2, s.EntStart())
	require.Equal(t, uint64(42), s.EntLabel())

	s.CloseEntity()
	require.False(t, s.EntityIsOpen())
	require.Equal(t, -1, s.EntStart())
}

func TestState_NER_IOBSequence(t *testing.T) {
	s := NewState(4)
	s.SetEntIOB(0, 2) // O
	s.SetEntIOB(1, 3) // B
	s.SetEntIOB(2, 1) // I
	s.SetEntIOB(3, 1) // I
	require.Equal(t, uint8(2), s.EntIOB(0))
	require.Equal(t, uint8(3), s.EntIOB(1))
	require.Equal(t, uint8(1), s.EntIOB(2))
	require.Equal(t, uint8(1), s.EntIOB(3))
	// Out-of-range returns 0 (missing) without panicking.
	require.Equal(t, uint8(0), s.EntIOB(-1))
	require.Equal(t, uint8(0), s.EntIOB(99))
}

func TestState_NER_RecordEntityClearsOpen(t *testing.T) {
	s := NewState(5)
	s.OpenEntity(1, 42)
	s.RecordEntity(1, 3, 42)
	require.False(t, s.EntityIsOpen(), "RecordEntity must clear the open register")
	ents := s.Entities()
	require.Len(t, ents, 1)
	require.Equal(t, 1, ents[0].Start)
	require.Equal(t, 3, ents[0].End)
	require.Equal(t, uint64(42), ents[0].Label)
}
