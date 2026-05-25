package parserinternals

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBiluo_MoveName(t *testing.T) {
	require.Equal(t, "O", BiluoMoveName(ActionOut, ""))
	require.Equal(t, "M", BiluoMoveName(ActionMissing, ""))
	require.Equal(t, "B-PERSON", BiluoMoveName(ActionBegin, "PERSON"))
	require.Equal(t, "I-PERSON", BiluoMoveName(ActionIn, "PERSON"))
	require.Equal(t, "L-PERSON", BiluoMoveName(ActionLast, "PERSON"))
	require.Equal(t, "U-PERSON", BiluoMoveName(ActionUnit, "PERSON"))
	// Empty-label B/I/L/U is the bare letter (unit-no-label is in moves).
	require.Equal(t, "U", BiluoMoveName(ActionUnit, ""))
}

func TestBiluo_LoadMoves_SMBundle(t *testing.T) {
	movesPath := filepath.Join("..", "..", "testdata", "models", "en_core_web_sm", "ner", "moves")
	if _, err := os.Stat(movesPath); err != nil {
		t.Skipf("sm ner/moves not present: %s", movesPath)
	}
	ts, err := LoadBiluoMoves(movesPath)
	require.NoError(t, err)
	// Verified via Python: en_core_web_sm has 74 BILUO moves
	// (18 each for B/I/L, 19 for U incl. "", 1 for O).
	require.Equal(t, 74, ts.NMoves)

	// OUT exists at the very end (action 5 sorts last).
	idx, ok := LookupBiluoClass(ts, ActionOut, "")
	require.True(t, ok)
	require.Equal(t, 73, idx, "OUT is the last (highest-class-index) action")

	// First B-action is the highest-freq entity type. From the moves dump,
	// that's ORG (freq 56516).
	idx, ok = LookupBiluoClass(ts, ActionBegin, "ORG")
	require.True(t, ok)
	require.Equal(t, 0, idx, "B-ORG is the highest-freq B action (class index 0)")
}

func TestBiluo_LoadMoves_MDBundle(t *testing.T) {
	movesPath := filepath.Join("..", "..", "testdata", "models", "en_core_web_md", "ner", "moves")
	if _, err := os.Stat(movesPath); err != nil {
		t.Skipf("md ner/moves not present: %s", movesPath)
	}
	ts, err := LoadBiluoMoves(movesPath)
	require.NoError(t, err)
	require.Equal(t, 74, ts.NMoves)
	// md ships the same labels as sm; the move ordering is identical.
	idx, ok := LookupBiluoClass(ts, ActionBegin, "ORG")
	require.True(t, ok)
	require.Equal(t, 0, idx)
}

func TestBiluoIsValid_BeginRequiresClosedEntity(t *testing.T) {
	s := NewState(3)
	// B-PERSON valid at the start (buffer length 3, no entity, label != 0).
	require.Equal(t, 1, BiluoIsValid(s, ActionBegin, "PERSON", 42, 0))
	s.OpenEntity(0, 42)
	// B-PERSON now invalid (entity open).
	require.Equal(t, 0, BiluoIsValid(s, ActionBegin, "PERSON", 42, 42))
}

func TestBiluoIsValid_OutOnlyWhenClosedAndBufferNonEmpty(t *testing.T) {
	s := NewState(2)
	require.Equal(t, 1, BiluoIsValid(s, ActionOut, "", 0, 0))
	s.OpenEntity(0, 42)
	require.Equal(t, 0, BiluoIsValid(s, ActionOut, "", 0, 42), "OUT invalid while entity open")
	s.CloseEntity()
	// Drain the buffer to empty.
	s.Push()
	s.Push()
	s.Pop()
	s.Pop()
	require.Equal(t, 0, BiluoIsValid(s, ActionOut, "", 0, 0), "OUT invalid with empty buffer")
}

func TestBiluoIsValid_BeginRequiresBufferLen2(t *testing.T) {
	// With only 1 token left, BEGIN is invalid (no room for I/L) — must U.
	s := NewState(1)
	require.Equal(t, 0, BiluoIsValid(s, ActionBegin, "PERSON", 42, 0))
	// UNIT IS valid on a single-token buffer.
	require.Equal(t, 1, BiluoIsValid(s, ActionUnit, "PERSON", 42, 0))
}

func TestBiluoIsValid_InRequiresLabelMatch(t *testing.T) {
	s := NewState(3)
	s.OpenEntity(0, 42) // open entity, label hash 42
	// I with matching label.
	require.Equal(t, 1, BiluoIsValid(s, ActionIn, "PERSON", 42, 42))
	// I with mismatched label.
	require.Equal(t, 0, BiluoIsValid(s, ActionIn, "OTHER", 99, 42))
}

func TestBiluoIsValid_LastRequiresLabelMatch(t *testing.T) {
	s := NewState(2)
	s.OpenEntity(0, 42)
	require.Equal(t, 1, BiluoIsValid(s, ActionLast, "PERSON", 42, 42))
	require.Equal(t, 0, BiluoIsValid(s, ActionLast, "OTHER", 99, 42))
	// LAST with no open entity is invalid.
	s.CloseEntity()
	require.Equal(t, 0, BiluoIsValid(s, ActionLast, "PERSON", 42, 0))
}

func TestBiluoApply_UnitClosesImmediately(t *testing.T) {
	s := NewState(3)
	BiluoApply(s, ActionUnit, "ORG", 99)
	require.False(t, s.EntityIsOpen(), "UNIT closes immediately")
	require.Equal(t, uint8(3), s.EntIOB(0), "UNIT writes B encoding")
	ents := s.Entities()
	require.Len(t, ents, 1)
	require.Equal(t, 0, ents[0].Start)
	require.Equal(t, 1, ents[0].End)
	require.Equal(t, uint64(99), ents[0].Label)
}

func TestBiluoApply_BeginThenLast(t *testing.T) {
	s := NewState(3)
	BiluoApply(s, ActionBegin, "PERSON", 42)
	require.True(t, s.EntityIsOpen())
	require.Equal(t, 0, s.EntStart())
	require.Equal(t, uint64(42), s.EntLabel())
	require.Equal(t, uint8(3), s.EntIOB(0))
	// Token 0 consumed; B(0) is now token 1.
	require.Equal(t, 1, s.B(0))

	BiluoApply(s, ActionLast, "PERSON", 42)
	require.False(t, s.EntityIsOpen())
	require.Equal(t, uint8(1), s.EntIOB(1), "LAST writes I encoding")
	ents := s.Entities()
	require.Len(t, ents, 1)
	require.Equal(t, 0, ents[0].Start)
	require.Equal(t, 2, ents[0].End, "half-open [0, 2)")
	require.Equal(t, uint64(42), ents[0].Label)
}

func TestBiluoApply_OutAdvancesBufferWithO(t *testing.T) {
	s := NewState(2)
	BiluoApply(s, ActionOut, "", 0)
	require.Equal(t, uint8(2), s.EntIOB(0), "OUT writes O encoding")
	require.False(t, s.EntityIsOpen())
	require.Equal(t, 1, s.B(0), "buffer advanced by 1")
	require.Empty(t, s.Entities(), "OUT does not record a span")
}

func TestBiluoSetValid_DispatchesPerRow(t *testing.T) {
	s := NewState(3)
	// Synthesize a tiny TransitionSystem: B-PERSON (class 0), O (class 1).
	ts := &TransitionSystem{
		Transitions: []Transition{
			{ClassIndex: 0, Move: ActionBegin, Label: "PERSON"},
			{ClassIndex: 1, Move: ActionOut, Label: ""},
		},
		NMoves:     2,
		classByKey: map[string]int{},
	}
	labelHashes := []uint64{42, 0}
	out := make([]int32, 2)
	BiluoSetValid(out, s, ts, labelHashes)
	require.Equal(t, int32(1), out[0], "B-PERSON valid at start")
	require.Equal(t, int32(1), out[1], "O valid at start")

	// Open an entity → B-PERSON invalid, O invalid (entity_is_open).
	s.OpenEntity(0, 42)
	BiluoSetValid(out, s, ts, labelHashes)
	require.Equal(t, int32(0), out[0])
	require.Equal(t, int32(0), out[1])
}
