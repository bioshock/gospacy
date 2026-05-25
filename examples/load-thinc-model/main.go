// Command load-thinc-model loads a Python-trained thinc Chain(Linear,Softmax)
// model and runs a forward pass on a small fixed input.
//
// Usage:
//
//	load-thinc-model <model.msgpack>
//
// Demonstrates: nn.Model construction, layers.Chain, layers.Linear,
// layers.Softmax, nn.Model.FromBytes, nn.Model.Predict. The model must be a
// Chain of Linear(3,4) + Softmax(4,2) — i.e., the same shape as the
// testdata/golden/tiny_thinc_model.msgpack from the gospacy test fixtures.
package main

import (
	"fmt"
	"os"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/bioshock/gospacy/v3/nn/layers"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: load-thinc-model <model.msgpack>")
		os.Exit(2)
	}
	path := os.Args[1]

	bytesData, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", path, err)
		os.Exit(1)
	}

	ops := gonum.New()
	linear := layers.Linear(ops, 4, 3)
	softmax := layers.Softmax(ops, 2, 4)
	model := layers.Chain(ops, linear, softmax)

	if err := model.FromBytes(bytesData); err != nil {
		fmt.Fprintf(os.Stderr, "FromBytes: %v\n", err)
		os.Exit(1)
	}

	X := nn.Floats2d{
		Data: []float32{
			0.1, 0.2, 0.3,
			-0.4, 0.5, -0.6,
			0.7, -0.8, 0.9,
		},
		Rows: 3, Cols: 3,
	}

	out, err := model.Predict(X)
	if err != nil {
		fmt.Fprintf(os.Stderr, "predict: %v\n", err)
		os.Exit(1)
	}

	y, ok := out.(nn.Floats2d)
	if !ok {
		fmt.Fprintf(os.Stderr, "unexpected output type %T\n", out)
		os.Exit(1)
	}

	fmt.Printf("Output shape: (%d, %d)\n", y.Rows, y.Cols)
	for i := 0; i < y.Rows; i++ {
		fmt.Printf("  row %d: %v\n", i, y.Data[i*y.Cols:(i+1)*y.Cols])
	}
}
