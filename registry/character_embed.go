package registry

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
)

// buildCharacterEmbedV2 implements spacy.CharacterEmbed.v2.
//
// Structural model with one 3-D parameter E of shape (nC, nV=256, nM) and
// output dimension nO=nC*nM. The forward pass is documented in
// upstream/spaCy/spacy/ml/_character_embed.py; the runtime helper that
// consumes the param lives alongside the Tagger feature pipeline (out of
// scope for Phase 4 because en_core_web_sm does not exercise it — kept here
// for registry completeness so bundles that DO use it stop being skipped).
func buildCharacterEmbedV2(cfg map[string]any) (*nn.Model, error) {
	nM := cfgInt(cfg, "nM", 0)
	nC := cfgInt(cfg, "nC", 0)
	if nM <= 0 || nC <= 0 {
		return nil, fmt.Errorf("CharacterEmbed.v2: nM and nC must be positive (got nM=%d nC=%d)", nM, nC)
	}
	ops := gonum.New()
	m := &nn.Model{Name: "charembed", Ops: ops, Attrs: map[string]any{}}
	m.Attrs["nM"] = int64(nM)
	m.Attrs["nC"] = int64(nC)
	m.Attrs["nV"] = int64(256)
	m.Attrs["nO"] = int64(nC * nM)
	return m, nil
}
