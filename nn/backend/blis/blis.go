//go:build blis

// Package blis is the cgo+BLIS backend, opt-in via `-tags blis`.
//
// Phase 1a status: scaffold only. The full nn.Ops implementation lands in
// Phase 1b. The build-tag wiring exists so `go build -tags blis ./...`
// succeeds, proving the build system works before cgo bindings are added.
package blis

import "errors"

// ErrNotImplemented is returned by every op until Phase 1b fills in the cgo bindings.
var ErrNotImplemented = errors.New("blis backend: not implemented in Phase 1a")

// Ops is a placeholder struct. The full nn.Ops interface implementation lands in Phase 1b.
type Ops struct{}

// New returns the cgo+BLIS Ops. In Phase 1a this is a stub.
func New() *Ops { return &Ops{} }
