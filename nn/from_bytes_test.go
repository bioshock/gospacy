package nn

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

func nnRepoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..")
}

// buildTinyChain returns the Go-side tree mirroring the Python chain in
// dump_thinc_model.py. The tree has 3 nodes in pre-order: root, linear, softmax.
// FromBytes will overwrite Names/Dims/Params from the payload.
func buildTinyChain() *Model {
	linear := &Model{
		Name:   "linear",
		Params: map[string][]float32{"W": nil, "b": nil},
		Dims:   map[string]int{},
	}
	softmax := &Model{
		Name:   "softmax",
		Params: map[string][]float32{"W": nil, "b": nil},
		Dims:   map[string]int{},
	}
	root := &Model{
		Name:   "linear>>softmax",
		Layers: []*Model{linear, softmax},
		Dims:   map[string]int{},
	}
	return root
}

func TestFromBytes_LoadsTinyChain(t *testing.T) {
	bytesPath := filepath.Join(nnRepoRoot(t), "testdata", "golden", "tiny_thinc_model.msgpack")
	bytesData, err := os.ReadFile(bytesPath)
	require.NoError(t, err, "run `testharness/.venv/bin/python testharness/dump_thinc_model.py` first")

	root := buildTinyChain()
	require.NoError(t, root.FromBytes(bytesData))

	// After loading, each leaf should have W and b populated to the right shape.
	linear := root.Layers[0]
	require.NotNil(t, linear.Params["W"])
	require.Len(t, linear.Params["W"], 4*3, "Linear W is (nO, nI) = (4, 3)")
	require.Len(t, linear.Params["b"], 4, "Linear b is (nO,) = (4,)")

	softmax := root.Layers[1]
	require.Len(t, softmax.Params["W"], 2*4, "Softmax W is (nO, nI) = (2, 4)")
	require.Len(t, softmax.Params["b"], 2, "Softmax b is (nO,) = (2,)")
}
