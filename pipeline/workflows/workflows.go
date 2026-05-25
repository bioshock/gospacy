// Package workflows holds end-to-end ports of representative real-world
// spaCy usage patterns (POS frequency, noun-chunk extraction, SVO triples,
// passive detection, sentence iteration, etc.). Each workflow is a pure
// function over a fully-piped *doc.Doc plus the bundle's StringStore; output
// is a JSON-marshalable value that gets diffed against a Python golden
// produced by the matching script under testharness/workflows/.
//
// Per-workflow files (wNN_<name>.go) export a Run function and the registry
// entry referencing it. The strict-100% differential is wired in
// workflows_real_test.go.
package workflows

import (
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// Workflow is one entry in the differential registry. Run is the pure Go
// implementation; GoldenPath is the path (relative to the test working dir,
// i.e. pipeline/workflows/) to the JSON file produced by the matching Python
// script under testharness/workflows/.
type Workflow struct {
	Name       string
	Run        func(d *doc.Doc, ss *vocab.StringStore) any
	GoldenPath string
}

// allWorkflows lists every workflow that participates in the strict
// Python-vs-Go differential. New workflows append themselves to this slice
// via their own init().
var allWorkflows []Workflow

// AllWorkflows returns the registry snapshot. Used by the differential test.
func AllWorkflows() []Workflow {
	out := make([]Workflow, len(allWorkflows))
	copy(out, allWorkflows)
	return out
}

// register is invoked from each wNN_*.go file's init() to append its
// Workflow to the package registry. Kept unexported so registration cannot
// leak outside the package.
func register(w Workflow) { allWorkflows = append(allWorkflows, w) }
