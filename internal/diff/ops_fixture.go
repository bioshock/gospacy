package diff

import (
	"encoding/json"
	"fmt"
)

// Array is a numpy-style float32 array dumped from Python as JSON.
type Array struct {
	Shape []int     `json:"shape"`
	Dtype string    `json:"dtype"`
	Data  []float32 `json:"data"`
}

// Uint32Array is for ops that output integer arrays (e.g., hash op).
type Uint32Array struct {
	Shape []int    `json:"shape"`
	Dtype string   `json:"dtype"`
	Data  []uint32 `json:"data"`
}

// OpsFixture is a per-op golden file produced by dump_ops.py write_op().
type OpsFixture struct {
	Op           string   `json:"op"`
	ThincVersion string   `json:"thinc_version"`
	Seed         int      `json:"seed"`
	Cases        []OpCase `json:"cases"`
}

// OpCase is one input/output pair for an op. Inputs and Output are kept as raw
// JSON so each op test can decode them with its own shape requirements
// (float32 for most ops, uint32 for hash, int32 for indices).
type OpCase struct {
	Name   string                     `json:"name"`
	Inputs map[string]json.RawMessage `json:"inputs"`
	Output json.RawMessage            `json:"output"`
	Extra  map[string]json.RawMessage `json:"extra,omitempty"`
}

// OpsSampleFixture is the in-repo `sample_ops.json` containing one tiny case
// per op, used by Go unit tests without re-running the full dumper.
type OpsSampleFixture struct {
	ThincVersion string            `json:"thinc_version"`
	Ops          map[string]OpCase `json:"ops"`
}

// LoadOpsFixture reads and decodes an OpsFixture from the JSON file at path.
func LoadOpsFixture(path string) (*OpsFixture, error) {
	var fx OpsFixture
	if err := loadJSON(path, &fx); err != nil {
		return nil, err
	}
	return &fx, nil
}

// LoadOpFixture reads and decodes an OpsSampleFixture from the JSON file at path.
func LoadOpFixture(path string) (*OpsSampleFixture, error) {
	var fx OpsSampleFixture
	if err := loadJSON(path, &fx); err != nil {
		return nil, err
	}
	return &fx, nil
}

// Float32Output decodes the case's Output as a float32 Array.
func (c *OpCase) Float32Output() (Array, error) {
	var a Array
	if err := json.Unmarshal(c.Output, &a); err != nil {
		return Array{}, fmt.Errorf("decode float32 output for case %q: %w", c.Name, err)
	}
	return a, nil
}

// Uint32Output decodes the case's Output as a uint32 Array (e.g., hash op).
func (c *OpCase) Uint32Output() (Uint32Array, error) {
	var a Uint32Array
	if err := json.Unmarshal(c.Output, &a); err != nil {
		return Uint32Array{}, fmt.Errorf("decode uint32 output for case %q: %w", c.Name, err)
	}
	return a, nil
}

// IntInput decodes a named input as an int (e.g., dimensions like m, k, n).
func (c *OpCase) IntInput(name string) (int, error) {
	raw, ok := c.Inputs[name]
	if !ok {
		return 0, fmt.Errorf("input %q not found in case %q", name, c.Name)
	}
	var n int
	if err := json.Unmarshal(raw, &n); err != nil {
		return 0, fmt.Errorf("decode int input %q: %w", name, err)
	}
	return n, nil
}

// ArrayInput decodes a named input as a float32 Array.
func (c *OpCase) ArrayInput(name string) (Array, error) {
	raw, ok := c.Inputs[name]
	if !ok {
		return Array{}, fmt.Errorf("input %q not found in case %q", name, c.Name)
	}
	var a Array
	if err := json.Unmarshal(raw, &a); err != nil {
		return Array{}, fmt.Errorf("decode array input %q: %w", name, err)
	}
	return a, nil
}

// IntsInput decodes a named input as an int slice (e.g., shape arrays).
func (c *OpCase) IntsInput(name string) ([]int, error) {
	raw, ok := c.Inputs[name]
	if !ok {
		return nil, fmt.Errorf("input %q not found in case %q", name, c.Name)
	}
	var v []int
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, fmt.Errorf("decode []int input %q: %w", name, err)
	}
	return v, nil
}
