package nn_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/bioshock/gospacy/v3/nn/layers"
)

// BenchmarkTinyChainForward times one forward pass of the loaded tiny thinc
// model (Linear(3,4) → Softmax(4,2)) on a 3-row input. This is the smallest
// meaningful end-to-end benchmark.
func BenchmarkTinyChainForward(b *testing.B) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	bytesPath := filepath.Join(root, "testdata", "golden", "tiny_thinc_model.msgpack")
	bytesData, err := os.ReadFile(bytesPath)
	if err != nil {
		b.Skip("tiny_thinc_model.msgpack missing; run testharness/.venv/bin/python testharness/dump_thinc_model.py first")
	}

	ops := gonum.New()
	linear := layers.Linear(ops, 4, 3)
	softmax := layers.Softmax(ops, 2, 4)
	model := layers.Chain(ops, linear, softmax)
	if err := model.FromBytes(bytesData); err != nil {
		b.Fatalf("FromBytes: %v", err)
	}

	X := nn.Floats2d{
		Data: []float32{
			0.1, 0.2, 0.3,
			-0.4, 0.5, -0.6,
			0.7, -0.8, 0.9,
		},
		Rows: 3, Cols: 3,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := model.Predict(X)
		if err != nil {
			b.Fatal(err)
		}
	}
}
