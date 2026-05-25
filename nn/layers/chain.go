package layers

import (
	"strings"

	"github.com/bioshock/gospacy/v3/nn"
)

// Chain composes sub-models sequentially. The output of one is fed to the next.
// Name format matches thinc: sub-layer names joined with ">>".
func Chain(ops nn.Ops, sublayers ...*nn.Model) *nn.Model {
	names := make([]string, len(sublayers))
	for i, s := range sublayers {
		names[i] = s.Name
	}
	return &nn.Model{
		Name:   strings.Join(names, ">>"),
		Ops:    ops,
		Layers: sublayers,
		Forward: func(m *nn.Model, X any) (any, error) {
			cur := X
			var err error
			for _, child := range m.Layers {
				cur, err = child.Predict(cur)
				if err != nil {
					return nil, err
				}
			}
			return cur, nil
		},
	}
}
