package parserinternals

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"

	"github.com/vmihailenco/msgpack/v5"
)

// Action IDs match the cdef enum in arc_eager.pyx (SHIFT=0, REDUCE=1, LEFT=2,
// RIGHT=3, BREAK=4). The order is load-bearing — LoadMoves walks action keys
// in numeric order and the model's n_moves rows assume that ordering.
const (
	ActionShift  = 0
	ActionReduce = 1
	ActionLeft   = 2
	ActionRight  = 3
	ActionBreak  = 4
)

// moveLetters mirrors MOVE_NAMES from arc_eager.pyx.
var moveLetters = [...]string{"S", "D", "L", "R", "B"}

// Transition is one (action, label) pair the parser may apply. ClassIndex is
// the position in TransitionSystem.Transitions and matches the output column
// of the upper Linear layer.
type Transition struct {
	ClassIndex int
	Move       int
	Label      string
}

// TransitionSystem owns the action × label lookup tables built from the
// bundle's parser/moves file. Inference-only: no strings table merge (we use
// the existing bundle Vocab), no oracle, no cost machinery.
type TransitionSystem struct {
	Transitions []Transition
	NMoves      int

	// classByKey indexes Transitions by "move/label" so LookupClass is O(1)
	// without a per-call linear scan.
	classByKey map[string]int
}

// MoveName mirrors arc_eager.move_name: e.g. "L-nsubj", "S", "B-ROOT".
func (ts *TransitionSystem) MoveName(classIndex int) string {
	tr := ts.Transitions[classIndex]
	if tr.Label == "" {
		return moveLetters[tr.Move]
	}
	return moveLetters[tr.Move] + "-" + tr.Label
}

// LookupClass returns the class index for (move, label), or (-1, false) when
// not present. Used by tests and by Parser.Apply when remapping a forced move
// (e.g. the BREAK fallback when sub-finalisation requires a sentence boundary).
func (ts *TransitionSystem) LookupClass(move int, label string) (int, bool) {
	idx, ok := ts.classByKey[transitionKey(move, label)]
	return idx, ok
}

func transitionKey(move int, label string) string {
	return moveLetters[move] + "/" + label
}

// movesEnvelope is the outer msgpack wrapper produced by
// TransitionSystem.to_bytes (transition_system.pyx:231).
type movesEnvelope struct {
	Moves string         `msgpack:"moves"`
	Cfg   map[string]any `msgpack:"cfg"`
}

// LoadMoves reads parser/moves and builds the TransitionSystem. The walk
// order must match Python's initialize_actions (transition_system.pyx:165)
// because the upper Linear's W has n_moves rows and column i corresponds to
// Transitions[i].
func LoadMoves(path string) (*TransitionSystem, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadMoves: %w", err)
	}
	var env movesEnvelope
	if err := msgpack.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("LoadMoves: msgpack decode: %w", err)
	}
	if env.Moves == "" {
		return nil, fmt.Errorf("LoadMoves: empty 'moves' field")
	}
	// env.Moves is JSON-encoded: {"<action_id>": {label: freq, ...}, ...}.
	var labelsByAction map[string]map[string]int
	if err := json.Unmarshal([]byte(env.Moves), &labelsByAction); err != nil {
		return nil, fmt.Errorf("LoadMoves: parse moves JSON: %w", err)
	}

	// Walk action keys in numeric-ascending order (Python: sorted(int(action))).
	actionKeys := make([]int, 0, len(labelsByAction))
	for k := range labelsByAction {
		n, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("LoadMoves: non-numeric action key %q", k)
		}
		actionKeys = append(actionKeys, n)
	}
	sort.Ints(actionKeys)

	ts := &TransitionSystem{classByKey: map[string]int{}}
	for _, action := range actionKeys {
		if action < 0 || action > ActionBreak {
			return nil, fmt.Errorf("LoadMoves: unknown action id %d", action)
		}
		labelFreqs := labelsByAction[strconv.Itoa(action)]

		// Build [(freq, label)] in Python's exact order:
		//   sorted_labels = [(f, L) for L, f in label_freqs.items()]
		//   sorted_labels.sort(); sorted_labels.reverse()
		// → ascending by (freq, label) then reversed → descending by (freq, label).
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
			ts.classByKey[transitionKey(action, e.Label)] = classIdx
		}
	}
	ts.NMoves = len(ts.Transitions)
	return ts, nil
}
