package parserinternals

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"

	"github.com/vmihailenco/msgpack/v5"
)

// BILUO action IDs. Order matches the cdef enum in
// spacy/pipeline/_parser_internals/ner.pyx. The numbers are load-bearing:
// the on-disk ner/moves msgpack keys actions by these IDs in string form
// ("0".."5") and the upper Linear's row ordering is built off them.
const (
	ActionMissing = 0
	ActionBegin   = 1
	ActionIn      = 2
	ActionLast    = 3
	ActionUnit    = 4
	ActionOut     = 5
)

// biluoLetters mirrors MOVE_NAMES from ner.pyx.
var biluoLetters = [...]string{"M", "B", "I", "L", "U", "O"}

// BiluoMoveName mirrors BiluoPushDown.move_name (ner.pyx:165). For OUT or
// MISSING we use the bare letter; for B/I/L/U we use "<letter>-<label>".
func BiluoMoveName(move int, label string) string {
	if move == ActionOut {
		return "O"
	}
	if move == ActionMissing {
		return "M"
	}
	if move < 0 || move >= len(biluoLetters) {
		return ""
	}
	if label == "" {
		return biluoLetters[move]
	}
	return biluoLetters[move] + "-" + label
}

// LoadBiluoMoves reads <bundle>/ner/moves and builds the BILUO
// TransitionSystem. Walk order mirrors initialize_actions
// (transition_system.pyx:165): action keys ascending; labels descending by
// (freq, label). The label "" (empty) appears under UNIT and OUT — same
// sort, sorts last within the action because freq=1 < any real entity-type
// freq. The walk must match upstream exactly: the upper Linear's W rows are
// indexed by ClassIndex which equals the position in this Transitions list.
func LoadBiluoMoves(path string) (*TransitionSystem, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadBiluoMoves: %w", err)
	}
	var env movesEnvelope
	if err := msgpack.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("LoadBiluoMoves: msgpack decode: %w", err)
	}
	if env.Moves == "" {
		return nil, fmt.Errorf("LoadBiluoMoves: empty 'moves' field")
	}
	var labelsByAction map[string]map[string]int
	if err := json.Unmarshal([]byte(env.Moves), &labelsByAction); err != nil {
		return nil, fmt.Errorf("LoadBiluoMoves: parse moves JSON: %w", err)
	}

	actionKeys := make([]int, 0, len(labelsByAction))
	for k := range labelsByAction {
		n, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("LoadBiluoMoves: non-numeric action key %q", k)
		}
		actionKeys = append(actionKeys, n)
	}
	sort.Ints(actionKeys)

	ts := &TransitionSystem{classByKey: map[string]int{}}
	for _, action := range actionKeys {
		if action < ActionMissing || action > ActionOut {
			return nil, fmt.Errorf("LoadBiluoMoves: unknown BILUO action id %d", action)
		}
		labelFreqs := labelsByAction[strconv.Itoa(action)]

		// Same descending-by-(freq, label) sort as ArcEager's LoadMoves —
		// upstream initialize_actions is shared between TS implementations.
		type freqLabel struct {
			Freq  int
			Label string
		}
		entries := make([]freqLabel, 0, len(labelFreqs))
		for lbl, f := range labelFreqs {
			entries = append(entries, freqLabel{Freq: f, Label: lbl})
		}
		sort.Slice(entries, func(i, j int) bool {
			if entries[i].Freq != entries[j].Freq {
				return entries[i].Freq > entries[j].Freq
			}
			return entries[i].Label > entries[j].Label
		})

		for _, e := range entries {
			classIdx := len(ts.Transitions)
			ts.Transitions = append(ts.Transitions, Transition{
				ClassIndex: classIdx,
				Move:       action,
				Label:      e.Label,
			})
			ts.classByKey[biluoTransitionKey(action, e.Label)] = classIdx
		}
	}
	ts.NMoves = len(ts.Transitions)
	return ts, nil
}

func biluoTransitionKey(move int, label string) string {
	return biluoLetters[move] + "/" + label
}

// LookupBiluoClass returns the class index for (move, label) in a BILUO
// TransitionSystem, or (-1, false) when not present. Mirrors
// TransitionSystem.LookupClass but uses the BILUO letter table so move IDs
// up to ActionOut (5) work; the inherited LookupClass on TransitionSystem
// only knows ArcEager's letters (S/D/L/R/B, 5 entries) and panics on
// move > 4.
func LookupBiluoClass(ts *TransitionSystem, move int, label string) (int, bool) {
	idx, ok := ts.classByKey[biluoTransitionKey(move, label)]
	return idx, ok
}

// BiluoIsValid returns 1 when (move, label) is a legal next action in s,
// else 0. Mirrors the per-class is_valid methods in ner.pyx (Begin / In /
// Last / Unit / Out). Notes vs upstream:
//
//   - Inference path only — preset_ent_iob is always 0 in our pipeline
//     (Token.EntIOB is zero before NER runs), so the preset_iob branches
//     collapse to their default arm.
//   - The IS_SPACE check on B_(0) is omitted: gospacy's tokenizer never
//     emits whitespace-only tokens (verified across the fixture corpus),
//     so the gate is unreachable in practice. If a future tokenizer change
//     produces whitespace tokens, this becomes a TODO.
//   - openLabel is the StringStore hash of the currently open entity's
//     label; passed in by the caller so we can compare without a string
//     lookup inside the hot loop.
func BiluoIsValid(s *State, move int, label string, labelHash, openLabel uint64) int {
	switch move {
	case ActionBegin:
		// Begin.is_valid (ner.pyx:362-400): no open entity, label != 0,
		// buffer_length >= 2 (we need B(1) to exist), B(1) not sent_start.
		if s.EntityIsOpen() {
			return 0
		}
		if s.BufferLength() < 2 {
			return 0
		}
		if labelHash == 0 {
			return 0
		}
		if s.IsSentStart(s.B(1)) == 1 {
			return 0
		}
		return 1
	case ActionIn:
		// In.is_valid (ner.pyx:440-475): open entity, buffer >= 2,
		// label matches open label, B(1) not sent_start, B(1) not the
		// start of another preset entity (preset_iob==3 — collapsed to
		// "no" since preset_iob is always 0).
		if !s.EntityIsOpen() {
			return 0
		}
		if s.BufferLength() < 2 {
			return 0
		}
		if labelHash == 0 || labelHash != openLabel {
			return 0
		}
		if s.IsSentStart(s.B(1)) == 1 {
			return 0
		}
		return 1
	case ActionLast:
		// Last.is_valid (ner.pyx:511-535): open entity, label matches.
		// buffer_length >= 1 is implicit (we have B(0) since !IsFinal).
		if !s.EntityIsOpen() {
			return 0
		}
		if labelHash == 0 || labelHash != openLabel {
			return 0
		}
		return 1
	case ActionUnit:
		// Unit.is_valid (ner.pyx:582-609): no open entity, label != 0,
		// buffer non-empty. The "label==0 only allowed for preset blocked"
		// branch is unreachable at inference (preset_iob==0).
		if s.EntityIsOpen() {
			return 0
		}
		if s.BufferLength() == 0 {
			return 0
		}
		if labelHash == 0 {
			return 0
		}
		return 1
	case ActionOut:
		// Out.is_valid (ner.pyx:647-658): no open entity, buffer non-empty.
		// The preset_iob==3 / preset_iob==1 branches collapse to "no" at
		// inference.
		if s.EntityIsOpen() {
			return 0
		}
		if s.BufferLength() == 0 {
			return 0
		}
		return 1
	}
	return 0
}

// BiluoSetValid populates out[i] = BiluoIsValid for each transition in ts.
// Mirrors BiluoPushDown's inherited set_valid (transition_system.pyx:152).
// openLabel is the open entity's label hash (0 when no entity is open) —
// pre-computed once per state to avoid an extra State method call per row.
func BiluoSetValid(out []int32, s *State, ts *TransitionSystem, labelHashes []uint64) {
	openLabel := s.EntLabel()
	for i, tr := range ts.Transitions {
		out[i] = int32(BiluoIsValid(s, tr.Move, tr.Label, labelHashes[i], openLabel))
	}
}

// BiluoApply mutates s for the (move, label) action. labelHash is the
// StringStore hash of label, resolved once by the caller. Mirrors the per-
// action `transition` methods in ner.pyx (Begin.transition etc.) which all
// invoke st.push()+st.pop() — that's the C++ way of advancing the buffer
// without leaving anything on the stack. The Push()+Pop() pair here is the
// faithful Go port: it consumes B(0) and leaves the stack empty.
//
// Writeback:
//   - OUT  → entIOB[B(0)] = 2 (O)
//   - BEGIN → entIOB[B(0)] = 3 (B), open the entity
//   - IN   → entIOB[B(0)] = 1 (I)
//   - LAST → entIOB[B(0)] = 1 (I; collapses to I at inference, matching
//     spaCy's set_annotations which does the same when ent_iob == 0)
//   - UNIT → entIOB[B(0)] = 3 (B for single-token entities)
//
// The Doc.set_ents path in spaCy's set_annotations writes ent_iob based on
// the recorded entity spans, not on internal L codes. We mirror that by
// keeping the entities slice (RecordEntity on UNIT and LAST) and treating
// entIOB as the per-token writeback Token.EntIOB will inherit.
func BiluoApply(s *State, move int, label string, labelHash uint64) {
	switch move {
	case ActionOut:
		if i := s.B(0); i >= 0 {
			s.SetEntIOB(i, 2) // O
		}
		s.Push()
		s.Pop()
	case ActionBegin:
		if i := s.B(0); i >= 0 {
			s.SetEntIOB(i, 3) // B
			s.OpenEntity(i, labelHash)
		}
		s.Push()
		s.Pop()
	case ActionIn:
		if i := s.B(0); i >= 0 {
			s.SetEntIOB(i, 1) // I
		}
		s.Push()
		s.Pop()
	case ActionLast:
		i := s.B(0)
		if i >= 0 {
			s.SetEntIOB(i, 1) // I (last token of multi-token entity)
		}
		start := s.EntStart()
		if start >= 0 && i >= 0 {
			s.RecordEntity(start, i+1, s.EntLabel())
		}
		s.Push()
		s.Pop()
	case ActionUnit:
		if i := s.B(0); i >= 0 {
			s.SetEntIOB(i, 3) // B for single-token entities
			s.RecordEntity(i, i+1, labelHash)
		}
		s.Push()
		s.Pop()
	}
}
