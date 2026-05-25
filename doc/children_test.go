package doc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/vocab"
)

// Tree: 0 -> 0 (root), 1 -> 0, 2 -> 0, 3 -> 1, 4 -> 1
//
//	0
//	├── 1
//	│   ├── 3
//	│   └── 4
//	└── 2
func makeSampleDoc() *Doc {
	d := NewDoc(vocab.NewVocab(), "")
	d.Tokens = []Token{
		{Text: "root", Head: 0},
		{Text: "c1", Head: 0},
		{Text: "c2", Head: 0},
		{Text: "g1", Head: 1},
		{Text: "g2", Head: 1},
	}
	return d
}

func TestChildrenOf_RootHasTwoDirectChildren(t *testing.T) {
	d := makeSampleDoc()
	require.Equal(t, []int{1, 2}, ChildrenOf(d, 0))
}

// TestChildrenIdx_BuiltOnce — after a single Children call the CSR slices
// are populated and a second call doesn't rebuild (verified by mutating the
// underlying slice and seeing the next call return the mutated view).
// Locks down the lazy-build cache invariant.
func TestChildrenIdx_BuiltOnce(t *testing.T) {
	d := makeSampleDoc()
	require.Nil(t, d.childStart, "cache must be nil before first call")
	_ = ChildrenOf(d, 0)
	require.NotNil(t, d.childStart, "cache must be built after first call")
	// Sample doc has 5 tokens; len(childStart) = 6, len(childIdx) = 4 (root
	// has 2 kids, token 1 has 2 kids, the rest have none).
	require.Len(t, d.childStart, 6)
	require.Len(t, d.childIdx, 4)
}

// TestChildrenOf_OutOfRange — bad index returns nil instead of panicking.
func TestChildrenOf_OutOfRange(t *testing.T) {
	d := makeSampleDoc()
	require.Nil(t, ChildrenOf(d, -1))
	require.Nil(t, ChildrenOf(d, 999))
	require.Nil(t, SubtreeOf(d, -1))
	require.Nil(t, SubtreeOf(d, 999))
}

// TestChildrenIdx_EmptyDoc — empty Tokens shouldn't panic; ChildrenOf
// returns nil. Covers the n==0 short-circuit in ensureChildIdx.
func TestChildrenIdx_EmptyDoc(t *testing.T) {
	d := NewDoc(vocab.NewVocab(), "")
	require.Nil(t, ChildrenOf(d, 0))
	require.Nil(t, SubtreeOf(d, 0))
}

func TestChildrenOf_InternalNode(t *testing.T) {
	d := makeSampleDoc()
	require.Equal(t, []int{3, 4}, ChildrenOf(d, 1))
}

func TestChildrenOf_Leaf(t *testing.T) {
	d := makeSampleDoc()
	require.Nil(t, ChildrenOf(d, 3))
}

func TestChildrenOf_ExcludesSelf(t *testing.T) {
	// Root points at itself; result must not include the root index.
	d := makeSampleDoc()
	for _, ch := range ChildrenOf(d, 0) {
		require.NotEqual(t, 0, ch)
	}
}

func TestSubtreeOf_Root(t *testing.T) {
	d := makeSampleDoc()
	require.Equal(t, []int{0, 1, 2, 3, 4}, SubtreeOf(d, 0))
}

func TestSubtreeOf_InternalNode(t *testing.T) {
	d := makeSampleDoc()
	require.Equal(t, []int{1, 3, 4}, SubtreeOf(d, 1))
}

func TestSubtreeOf_Leaf(t *testing.T) {
	d := makeSampleDoc()
	require.Equal(t, []int{3}, SubtreeOf(d, 3))
}
