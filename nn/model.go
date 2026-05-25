package nn

// ForwardFunc runs a model's forward pass. Inference-only: no is_train flag,
// no backprop callback. The input and output are `any` because thinc layers
// operate on heterogeneous types (Floats2d, Ints1d, Ragged, Padded, FloatList).
// Each layer's Forward type-asserts the expected input kind.
type ForwardFunc func(m *Model, X any) (any, error)

// Model is a node in the model tree. Children are in Layers; weights in Params;
// integer sizes in Dims; arbitrary state in Attrs; references to other tree
// nodes (by name) in Refs.
type Model struct {
	Name    string
	Layers  []*Model
	Params  map[string][]float32
	Dims    map[string]int
	Attrs   map[string]any
	Refs    map[string]*Model
	Ops     Ops
	Forward ForwardFunc
}

// Walk returns the model tree in breadth-first order, matching thinc's
// `Model.walk()` default (`order="bfs"`). The order is significant because
// `FromBytes` indexes the on-disk payload's parallel `nodes`/`params`/`attrs`
// arrays by walk position, and thinc's `Model.to_bytes()` serialises them in
// BFS order.
func (m *Model) Walk() []*Model {
	out := make([]*Model, 0, 1)
	queue := []*Model{m}
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		out = append(out, node)
		queue = append(queue, node.Layers...)
	}
	return out
}

// Predict runs forward inference on X, returning the model output.
func (m *Model) Predict(X any) (any, error) {
	if m.Forward == nil {
		return X, nil
	}
	return m.Forward(m, X)
}

// ---- Tensor kinds ----

// Floats2d is a row-major (Rows, Cols) float32 matrix.
type Floats2d struct {
	Data []float32
	Rows int
	Cols int
}

// Floats3d is a row-major (D0, D1, D2) float32 tensor.
type Floats3d struct {
	Data []float32
	D0   int
	D1   int
	D2   int
}

// Ints1d is a 1D int32 array (used for token feature IDs).
type Ints1d struct {
	Data []int32
}

// Ints2d is a row-major (Rows, Cols) int32 matrix.
type Ints2d struct {
	Data []int32
	Rows int
	Cols int
}

// Uint64s1d is a 1D uint64 array (HashEmbed inputs).
type Uint64s1d struct {
	Data []uint64
}

// Uint64s2d is a row-major (Rows, Cols) uint64 matrix. ExtractFeatures emits a
// list of these (one per Doc); IntsGetitem slices a single column out to feed
// HashEmbed.
type Uint64s2d struct {
	Data []uint64
	Rows int
	Cols int
}

// Ragged is a concatenated batch of variable-length 2D sequences, all with the
// same Cols. Lengths is per-sequence row count; sum(Lengths) == len(Data)/Cols.
type Ragged struct {
	Data    []float32
	Lengths []int32
	Cols    int
}

// RaggedU64 is the uint64 analogue of Ragged. Produced by List2Ragged on the
// FeatureExtractor → MultiHashEmbed path: ExtractFeatures emits a
// []Uint64s2d (one Uint64s2d per Doc), List2Ragged concatenates row-wise into
// a single RaggedU64. WithArray's Forward dispatches on this type to feed the
// inner MultiHashEmbed graph (which expects Uint64s2d, not Floats2d).
type RaggedU64 struct {
	Data    []uint64
	Lengths []int32
	Cols    int
}

// Padded is a time-major (T, B, W) tensor with per-timestep size bookkeeping.
// SizeAtT[t] is the number of sequences alive at timestep t (== count of
// Lengths >= t+1). Indices records the sort order (sequences are sorted
// length-desc internally; Indices maps sorted-order → original-order).
type Padded struct {
	Data    []float32
	SizeAtT []int32
	Lengths []int32
	Indices []int32
	B       int
	T       int
	W       int
}

// FloatList is a list of 2D float arrays (for thinc's List[Floats2d]).
type FloatList struct {
	Items []Floats2d
}
