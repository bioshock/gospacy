// Package pipeline holds the rule-based and neural components that mutate a
// Doc after tokenization: Tagger, AttributeRuler, Lemmatizer (and later
// Parser, NER). Each component exposes Apply(d *doc.Doc, ...) error.
package pipeline

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/nn"
)

// Tagger writes Token.Tag for every token in a Doc by running the bundle's
// tagger model on a pre-computed tok2vec output and argmax-ing per token.
// The fine→coarse POS mapping is the AttributeRuler's job, not the Tagger's.
//
// Apply takes the tok2vec output as a parameter (rather than invoking the
// listener's upstream chain itself) so that Bundle.Pipe can compute the
// tok2vec forward once and feed it to both Tagger and Parser.
type Tagger struct {
	model  *nn.Model
	labels []string
	src    BundleSource
}

// NewTagger constructs a Tagger from a loaded bundle. Requires:
//   - b.PipeLookup("tagger") returns a non-nil, non-skipped model
//   - tagger/cfg present on disk with a non-empty "labels" array
func NewTagger(b BundleSource) (*Tagger, error) {
	model, skipped, skipReason, ok := b.PipeLookup("tagger")
	if !ok {
		return nil, fmt.Errorf("Tagger: bundle has no 'tagger' pipe")
	}
	if skipped || model == nil {
		return nil, fmt.Errorf("Tagger: pipe skipped (%s)", skipReason)
	}
	cfgPath := filepath.Join(b.BundlePath(), "tagger", "cfg")
	raw, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, fmt.Errorf("Tagger: read tagger/cfg: %w", err)
	}
	var taggerCfg struct {
		Labels []string `json:"labels"`
	}
	if err := json.Unmarshal(raw, &taggerCfg); err != nil {
		return nil, fmt.Errorf("Tagger: parse tagger/cfg: %w", err)
	}
	if len(taggerCfg.Labels) == 0 {
		return nil, fmt.Errorf("Tagger: tagger/cfg has empty labels")
	}
	// Intern labels into the shared StringStore so Apply can resolve hashes
	// later without re-Adding on every call.
	for _, lbl := range taggerCfg.Labels {
		b.BundleVocab().StringStore().Add(lbl)
	}
	return &Tagger{model: model, labels: taggerCfg.Labels, src: b}, nil
}

// NewTaggerFromModel constructs a Tagger directly from a hand-built model and
// label list — the synthetic-weights unit test entry point. The vocab is the
// StringStore owner that receives labels via Add and is read by Apply.
func NewTaggerFromModel(model *nn.Model, labels []string, b BundleSource) *Tagger {
	if b != nil {
		for _, lbl := range labels {
			b.BundleVocab().StringStore().Add(lbl)
		}
	}
	return &Tagger{model: model, labels: labels, src: b}
}

// Apply runs the tagger over a single Doc using a pre-computed tok2vec
// output, writing Token.Tag (interned via d.Vocab.StringStore).
//
// tok2vecOutput must have one row per token (Rows == d.NumTokens()) and Cols
// equal to the model's input width (the listener's nO). The function pushes
// the rows through the model's `with_array(softmax)` sub-layer (Layers[1] of
// the 4-node tree built by buildTaggerV2) wrapped as a single Ragged
// sequence, then argmax per row → labels[i] → token.Tag.
func (t *Tagger) Apply(d *doc.Doc, tok2vecOutput nn.Floats2d) error {
	if d.NumTokens() == 0 {
		return nil
	}
	if tok2vecOutput.Rows != d.NumTokens() {
		return fmt.Errorf("Tagger.Apply: tok2vec rows %d != tokens %d",
			tok2vecOutput.Rows, d.NumTokens())
	}
	if len(t.model.Layers) < 2 {
		return fmt.Errorf("Tagger.Apply: model has %d sub-layers, expected >= 2 (Chain(listener, with_array(softmax)))",
			len(t.model.Layers))
	}
	// model.Layers[0] is the tok2vec listener (no params, passthrough).
	// model.Layers[1] is `with_array(softmax)` — the layer we run here.
	withArray := t.model.Layers[1]
	ragged := nn.Ragged{
		Data:    tok2vecOutput.Data,
		Lengths: []int32{int32(d.NumTokens())},
		Cols:    tok2vecOutput.Cols,
	}
	raw, err := withArray.Predict(ragged)
	if err != nil {
		return fmt.Errorf("Tagger.Apply: predict: %w", err)
	}
	scores, ok := raw.(nn.Ragged)
	if !ok {
		return fmt.Errorf("Tagger.Apply: predict returned %T, want Ragged", raw)
	}
	totalRows := 0
	for _, l := range scores.Lengths {
		totalRows += int(l)
	}
	if totalRows != d.NumTokens() {
		return fmt.Errorf("Tagger.Apply: scores rows %d != tokens %d", totalRows, d.NumTokens())
	}
	if scores.Cols != len(t.labels) {
		return fmt.Errorf("Tagger.Apply: scores cols %d != labels %d", scores.Cols, len(t.labels))
	}
	ss := d.Vocab.StringStore()
	for i := 0; i < d.NumTokens(); i++ {
		row := scores.Data[i*scores.Cols : (i+1)*scores.Cols]
		best := 0
		for j := 1; j < len(row); j++ {
			if row[j] > row[best] {
				best = j
			}
		}
		d.Tokens[i].Tag = ss.Add(t.labels[best])
	}
	return nil
}
