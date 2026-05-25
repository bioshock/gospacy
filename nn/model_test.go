package nn

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestModel_Walk_SingleNode(t *testing.T) {
	m := &Model{Name: "leaf"}
	walked := m.Walk()
	require.Len(t, walked, 1)
	require.Equal(t, "leaf", walked[0].Name)
}

func TestModel_Walk_BreadthFirst(t *testing.T) {
	// Tree:
	//   root
	//   ├─ A
	//   │  └─ A1
	//   └─ B
	//
	// BFS visits siblings before grandchildren, matching thinc's `Model.walk()`
	// default order which dictates the layout of `to_bytes()` and the inverse
	// `FromBytes` indexing.
	a1 := &Model{Name: "A1"}
	a := &Model{Name: "A", Layers: []*Model{a1}}
	b := &Model{Name: "B"}
	root := &Model{Name: "root", Layers: []*Model{a, b}}

	walked := root.Walk()
	require.Equal(t, []string{"root", "A", "B", "A1"}, namesOf(walked))
}

func TestFloats2d_Shape(t *testing.T) {
	x := Floats2d{Data: []float32{1, 2, 3, 4, 5, 6}, Rows: 2, Cols: 3}
	require.Equal(t, 2, x.Rows)
	require.Equal(t, 3, x.Cols)
	require.Len(t, x.Data, 6)
}

func TestRagged_Shape(t *testing.T) {
	r := Ragged{Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}, Lengths: []int32{2, 2}, Cols: 2}
	require.Equal(t, 2, int(r.Lengths[0]))
	require.Equal(t, 4, len(r.Data)/r.Cols) // total rows == 4
}

func namesOf(ms []*Model) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.Name
	}
	return out
}
