package registry

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
)

// buildTok2VecListenerV1 implements spacy.Tok2VecListener.v1 as an inference
// proxy. The listener has no parameters of its own; at predict time the
// pipeline runner (bundle.Pipe) supplies the cached tok2vec output computed by
// the upstream component named in Attrs["upstream"].
//
// The width is recorded for runtime sanity-checks (Tagger expects tok2vec
// output Cols == width).
func buildTok2VecListenerV1(cfg map[string]any) (*nn.Model, error) {
	upstream, ok := cfg["upstream"].(string)
	if !ok || upstream == "" {
		return nil, fmt.Errorf("Tok2VecListener.v1: missing or empty upstream")
	}
	width, ok := cfg["width"].(int64)
	if !ok {
		// Defensive: config parses both int64 and float64 by type promotion.
		if f, fok := cfg["width"].(float64); fok {
			width = int64(f)
			ok = true
		}
	}
	if !ok {
		return nil, fmt.Errorf("Tok2VecListener.v1: missing width")
	}
	ops := gonum.New()
	m := &nn.Model{Name: "tok2vec_listener", Ops: ops, Attrs: map[string]any{}}
	m.Attrs["upstream"] = upstream
	m.Attrs["width"] = width
	return m, nil
}
