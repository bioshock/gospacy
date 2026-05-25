package pipeline

import (
	"github.com/bioshock/gospacy/v3/config"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/vocab"
)

// BundleSource is the minimal interface the pipeline constructors need from a
// loaded bundle. *bundle.Bundle satisfies this interface, allowing bundle to
// import pipeline without creating an import cycle.
//
// PipeLookup returns (model, skipped, skipReason, present). If present is
// false the named pipe does not exist in the bundle at all.
type BundleSource interface {
	BundlePath() string
	BundleVocab() *vocab.Vocab
	BundleConfig() *config.Config
	PipeLookup(name string) (model *nn.Model, skipped bool, skipReason string, ok bool)
}
