package nn_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/bioshock/gospacy/v3/nn/layers"
	"github.com/stretchr/testify/require"
)

type ioPayload struct {
	ThincVersion     string            `json:"thinc_version"`
	ModelDescription string            `json:"model_description"`
	Dims             map[string]int    `json:"dims"`
	Input            golden            `json:"input"`
	LayerOutputs     map[string]golden `json:"layer_outputs"`
	FinalOutput      golden            `json:"final_output"`
}

type golden struct {
	Shape []int     `json:"shape"`
	Dtype string    `json:"dtype"`
	Data  []float32 `json:"data"`
}

func e2eRoot(t *testing.T) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..")
}

func loadIO(t *testing.T) *ioPayload {
	t.Helper()
	path := filepath.Join(e2eRoot(t), "testdata", "golden", "tiny_thinc_model_io.json")
	b, err := os.ReadFile(path)
	require.NoError(t, err)
	var p ioPayload
	require.NoError(t, json.Unmarshal(b, &p))
	return &p
}

func loadBytes(t *testing.T) []byte {
	t.Helper()
	path := filepath.Join(e2eRoot(t), "testdata", "golden", "tiny_thinc_model.msgpack")
	b, err := os.ReadFile(path)
	require.NoError(t, err, "run testharness/.venv/bin/python testharness/dump_thinc_model.py first")
	return b
}

func TestEndToEnd_TinyChainForwardPass(t *testing.T) {
	io := loadIO(t)
	require.Equal(t, "chain(Linear(3, 4), Softmax_v2(4, 2))", io.ModelDescription)

	ops := gonum.New()
	nI := io.Dims["nI"]
	nH := io.Dims["nHidden"]
	nO := io.Dims["nO"]

	linear := layers.Linear(ops, nH, nI)
	softmax := layers.Softmax(ops, nO, nH)
	root := layers.Chain(ops, linear, softmax)

	require.NoError(t, root.FromBytes(loadBytes(t)))

	require.Len(t, linear.Params["W"], nH*nI)
	require.Len(t, linear.Params["b"], nH)

	rows := io.Input.Shape[0]
	x := nn.Floats2d{Data: io.Input.Data, Rows: rows, Cols: nI}

	linearOut, err := linear.Predict(x)
	require.NoError(t, err)
	yLin := linearOut.(nn.Floats2d)
	want := io.LayerOutputs["linear"]
	rep := diff.CompareFloats(want.Data, yLin.Data, diff.Tolerance{AbsMax: 1e-5, RelMax: 1e-5})
	require.Truef(t, rep.Equal(),
		"Linear output mismatch: first disagree at %d, maxAbsDiff=%g",
		rep.FirstDisagreeIdx, rep.MaxAbsDiff)

	softOut, err := softmax.Predict(yLin)
	require.NoError(t, err)
	ySoft := softOut.(nn.Floats2d)
	want = io.LayerOutputs["softmax"]
	rep = diff.CompareFloats(want.Data, ySoft.Data, diff.Tolerance{AbsMax: 1e-5, RelMax: 1e-5})
	require.Truef(t, rep.Equal(),
		"Softmax output mismatch: first disagree at %d, maxAbsDiff=%g",
		rep.FirstDisagreeIdx, rep.MaxAbsDiff)

	finalOut, err := root.Predict(x)
	require.NoError(t, err)
	yFinal := finalOut.(nn.Floats2d)
	want = io.FinalOutput
	rep = diff.CompareFloats(want.Data, yFinal.Data, diff.Tolerance{AbsMax: 1e-5, RelMax: 1e-5})
	require.Truef(t, rep.Equal(),
		"Chain output mismatch: first disagree at %d, maxAbsDiff=%g",
		rep.FirstDisagreeIdx, rep.MaxAbsDiff)
}
