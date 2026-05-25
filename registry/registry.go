// Package registry maps @architectures strings (from config.cfg) to Go
// factory functions that build a *nn.Model tree. Mirrors spaCy's
// spacy/registrations.py + spacy-legacy package. Each name is either:
//   - Registered + implemented: factory returns a real *nn.Model.
//   - Registered + stub: factory returns ErrArchitectureNotImplemented with
//     a phase-N hint.
//   - Unregistered: Build returns ErrUnknownArchitecture.
package registry

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// ArchitectureFactory builds a *nn.Model from a hyperparameter map keyed by
// config.cfg key names (e.g. "width", "depth", "nP", "embed_size").
type ArchitectureFactory func(cfg map[string]any) (*nn.Model, error)

// ErrUnknownArchitecture signals an architecture not present in the registry.
type ErrUnknownArchitecture struct{ Name string }

func (e *ErrUnknownArchitecture) Error() string {
	return fmt.Sprintf("registry: unknown architecture %q (not in spacy.* or spacy-legacy.* namespaces)", e.Name)
}

// ErrArchitectureNotImplemented signals an architecture registered but stubbed.
type ErrArchitectureNotImplemented struct {
	Name  string
	Phase string
}

func (e *ErrArchitectureNotImplemented) Error() string {
	return fmt.Sprintf("registry: %q registered but not implemented (planned: %s)", e.Name, e.Phase)
}

var registered = map[string]ArchitectureFactory{}

// Register installs a factory under name. Panics on duplicate (programmer error).
func Register(name string, fn ArchitectureFactory) {
	if _, ok := registered[name]; ok {
		panic(fmt.Sprintf("registry: duplicate registration for %q", name))
	}
	registered[name] = fn
}

// Build resolves name and invokes the factory with cfg.
func Build(name string, cfg map[string]any) (*nn.Model, error) {
	fn, ok := registered[name]
	if !ok {
		return nil, &ErrUnknownArchitecture{Name: name}
	}
	return fn(cfg)
}

// Names returns every registered architecture name (for diagnostics).
func Names() []string {
	out := make([]string, 0, len(registered))
	for n := range registered {
		out = append(out, n)
	}
	return out
}
