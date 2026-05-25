package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
)

// Dropout: inference-identity layer. thinc's Dropout.v1 short-circuits to a
// passthrough whenever rate==0 or is_train is false; gospacy is inference-only
// (is_train always false), so the forward is always identity.
//
// We still type-assert Floats2d so a mis-wired tree fails loud, and we keep
// dropout_rate + is_enabled in Attrs so FromBytes round-trips them (the real
// bundle's attrs[22] = {dropout_rate: 0.1, is_enabled: true}).
func Dropout(ops nn.Ops, rate float32) *nn.Model {
	return &nn.Model{
		Name: "dropout",
		Ops:  ops,
		Attrs: map[string]any{
			"dropout_rate": rate,
			"is_enabled":   true,
		},
		Forward: func(m *nn.Model, X any) (any, error) {
			x, ok := X.(nn.Floats2d)
			if !ok {
				return nil, fmt.Errorf("Dropout: expected Floats2d, got %T", X)
			}
			return x, nil
		},
	}
}
