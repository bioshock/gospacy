package parserinternals

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestLoadMoves_BundlePayload loads the real en_core_web_sm parser/moves and
// verifies n_moves, the action mix, and the well-known SHIFT/BREAK-ROOT entries.
func TestLoadMoves_BundlePayload(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "models", "en_core_web_sm", "parser", "moves")
	if _, err := os.Stat(path); err != nil {
		t.Skip("en_core_web_sm not present; run testharness/download_assets.sh")
	}
	moves, err := LoadMoves(path)
	require.NoError(t, err)

	// 1 SHIFT + 1 REDUCE + 47 LEFT + 56 RIGHT + 1 BREAK = 106
	require.Equal(t, 106, moves.NMoves)

	// First two transitions must be SHIFT-"" and REDUCE-"" because they're the
	// only entries under actions 0 and 1, and the action loop iterates sorted
	// keys.
	require.Equal(t, ActionShift, moves.Transitions[0].Move)
	require.Equal(t, "", moves.Transitions[0].Label)
	require.Equal(t, "S", moves.MoveName(0))

	require.Equal(t, ActionReduce, moves.Transitions[1].Move)
	require.Equal(t, "D", moves.MoveName(1))

	// Last action class is BREAK-ROOT (action 4 has exactly one label "ROOT").
	last := moves.Transitions[len(moves.Transitions)-1]
	require.Equal(t, ActionBreak, last.Move)
	require.Equal(t, "ROOT", last.Label)
	require.Equal(t, "B-ROOT", moves.MoveName(moves.NMoves-1))

	// LEFT and RIGHT counts.
	counts := map[int]int{}
	for _, tr := range moves.Transitions {
		counts[tr.Move]++
	}
	require.Equal(t, 1, counts[ActionShift])
	require.Equal(t, 1, counts[ActionReduce])
	require.Equal(t, 47, counts[ActionLeft])
	require.Equal(t, 56, counts[ActionRight])
	require.Equal(t, 1, counts[ActionBreak])

	// LookupClass round-trips.
	idx, ok := moves.LookupClass(ActionBreak, "ROOT")
	require.True(t, ok)
	require.Equal(t, moves.NMoves-1, idx)
}
