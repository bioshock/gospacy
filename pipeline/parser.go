package pipeline

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/pipeline/parserinternals"
)

// Parser is the dependency-parser pipe. NewParser loads parser/cfg and
// parser/moves; the model tree lives on the bundle's pipe registry (built by
// registry.spacy.TransitionBasedParser.v2 + FromBytes).
//
// Apply runs a greedy ArcEager parse over a single Doc, given the pre-computed
// tok2vec output. Writes Token.Head / Token.Dep / Token.SentStart.
//
// Inference only — greedy decoding (beam_width=1). Beam search, oracle, and
// learn_tokens are out of scope; see NOT_YET_PORTED.md.
type Parser struct {
	moves       *parserinternals.TransitionSystem
	labelHashes []uint64 // labelHashes[i] is StringStore hash of moves.Transitions[i].Label (0 for "")
	rootLabel   uint64   // hash of "ROOT"
	model       *nn.Model
	cfg         parserCfg
	src         BundleSource
}

type parserCfg struct {
	BeamWidth         int     `json:"beam_width"`
	BeamDensity       float64 `json:"beam_density"`
	LearnTokens       bool    `json:"learn_tokens"`
	MinActionFreq     int     `json:"min_action_freq"`
	IncorrectSpansKey *string `json:"incorrect_spans_key"`
}

// NewParser loads moves + cfg from <bundle>/parser/{moves,cfg}, interns the
// move labels into the bundle Vocab, and binds the model. Returns an error
// when beam_width > 1 (beam search is not supported — see NOT_YET_PORTED.md).
func NewParser(b BundleSource) (*Parser, error) {
	model, skipped, skipReason, ok := b.PipeLookup("parser")
	if !ok {
		return nil, fmt.Errorf("Parser: bundle has no 'parser' pipe")
	}
	if skipped || model == nil {
		return nil, fmt.Errorf("Parser: pipe skipped (%s)", skipReason)
	}

	cfgPath := filepath.Join(b.BundlePath(), "parser", "cfg")
	cfgBytes, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, fmt.Errorf("Parser: read parser/cfg: %w", err)
	}
	var cfg parserCfg
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return nil, fmt.Errorf("Parser: parse parser/cfg: %w", err)
	}
	if cfg.BeamWidth > 1 {
		return nil, fmt.Errorf("Parser: beam_width=%d not supported (greedy decode only)", cfg.BeamWidth)
	}
	if cfg.LearnTokens {
		return nil, fmt.Errorf("Parser: learn_tokens=true not supported (inference only)")
	}

	movesPath := filepath.Join(b.BundlePath(), "parser", "moves")
	ts, err := parserinternals.LoadMoves(movesPath)
	if err != nil {
		return nil, fmt.Errorf("Parser: load moves: %w", err)
	}

	ss := b.BundleVocab().StringStore()
	labelHashes := make([]uint64, ts.NMoves)
	for i, tr := range ts.Transitions {
		if tr.Label == "" {
			labelHashes[i] = 0
		} else {
			labelHashes[i] = ss.Add(tr.Label)
		}
	}
	rootLabel := ss.Add("ROOT")

	return &Parser{
		moves:       ts,
		labelHashes: labelHashes,
		rootLabel:   rootLabel,
		model:       model,
		cfg:         cfg,
		src:         b,
	}, nil
}

// ApplyStub is a degenerate Apply used by unit tests that don't want to run
// the full scorer: marks every token as a root with dep=rootHash. Production
// callers use Apply.
func (p *Parser) ApplyStub(d *doc.Doc, rootHash uint64) error {
	for i := range d.Tokens {
		d.Tokens[i].Head = i
		d.Tokens[i].Dep = rootHash
	}
	return nil
}

// Apply runs a greedy ArcEager parse over d, given the tok2vec output
// (T rows × 96 cols, the upstream tok2vec width). Writes Token.Head /
// Token.Dep / Token.SentStart and then deprojectivizes labels containing "||".
//
// On a parse failure (arg_max_if_valid returns -1), the loop forces the state
// final and emits self-headed ROOT arcs for any token without a recorded head
// — matching upstream's force_final behaviour.
func (p *Parser) Apply(d *doc.Doc, tok2vecOutput nn.Floats2d) error {
	if d.NumTokens() == 0 {
		return nil
	}
	T := d.NumTokens()
	if tok2vecOutput.Rows != T {
		return fmt.Errorf("Parser.Apply: tok2vec rows %d != tokens %d", tok2vecOutput.Rows, T)
	}

	// Step 1: project upstream tok2vec output (96 cols) → 64 cols through
	// the parser model's tok2vec ref Chain. The Chain is
	// Layers[0] = [listener, list2array, linear(96→64)]; we skip the
	// listener (it's a passthrough proxy that re-uses the upstream
	// tok2vec, matching the tagger's pattern) and feed the cached output
	// through list2array → linear.
	tok2vecRef := p.model.Layers[0]
	if len(tok2vecRef.Layers) < 3 {
		return fmt.Errorf("Parser.Apply: tok2vec ref has %d children, expected 3", len(tok2vecRef.Layers))
	}
	listInput := nn.FloatList{Items: []nn.Floats2d{tok2vecOutput}}
	afterList2Array, err := tok2vecRef.Layers[1].Predict(listInput)
	if err != nil {
		return fmt.Errorf("Parser.Apply: list2array: %w", err)
	}
	projected, err := tok2vecRef.Layers[2].Predict(afterList2Array)
	if err != nil {
		return fmt.Errorf("Parser.Apply: projection linear: %w", err)
	}
	X, ok := projected.(nn.Floats2d)
	if !ok {
		return fmt.Errorf("Parser.Apply: projection produced %T, want Floats2d", projected)
	}

	// Step 2: precompute the lower (PrecomputableAffine) cache.
	lower := p.model.Refs["lower"]
	if lower == nil {
		return fmt.Errorf("Parser.Apply: model.Refs[\"lower\"] is nil")
	}
	lowerRaw, err := lower.Predict(X)
	if err != nil {
		return fmt.Errorf("Parser.Apply: lower: %w", err)
	}
	lowerOut, ok := lowerRaw.(nn.Floats2d)
	if !ok {
		return fmt.Errorf("Parser.Apply: lower produced %T, want Floats2d", lowerRaw)
	}
	nF := lower.Dims["nF"]
	nO := lower.Dims["nO"]
	nP := lower.Dims["nP"]
	if nF != 8 {
		return fmt.Errorf("Parser.Apply: nF=%d, expected 8 (extra_state_tokens=false only)", nF)
	}
	if lowerOut.Rows != (T+1)*nF || lowerOut.Cols != nO*nP {
		return fmt.Errorf("Parser.Apply: lower output shape mismatch (rows=%d cols=%d, want %d×%d)",
			lowerOut.Rows, lowerOut.Cols, (T+1)*nF, nO*nP)
	}

	upper := p.model.Refs["upper"]
	if upper == nil {
		return fmt.Errorf("Parser.Apply: model.Refs[\"upper\"] is nil")
	}
	upperW := upper.Params["W"]
	upperB := upper.Params["b"]
	nMoves := upper.Dims["nO"]
	if nMoves != p.moves.NMoves {
		return fmt.Errorf("Parser.Apply: upper.nO=%d != moves.NMoves=%d", nMoves, p.moves.NMoves)
	}
	lowerBias := lower.Params["b"]
	if len(lowerBias) != nO*nP {
		return fmt.Errorf("Parser.Apply: lower.b length %d != nO*nP %d", len(lowerBias), nO*nP)
	}

	// Step 3: greedy decode.
	state := parserinternals.NewState(T)
	isValid := make([]int32, p.moves.NMoves)
	scores := make([]float32, p.moves.NMoves)
	var idsArr [8]int32
	ids := idsArr[:]
	for !state.IsFinal() {
		state.SetContextTokens8(&idsArr)
		ScoreStateInternal(scores, ids, nF, lowerOut.Data, lowerBias, upperW, upperB, nO, nP)
		parserinternals.SetValid(isValid, state, p.moves)
		guess := argMaxIfValid(scores, isValid)
		if guess < 0 {
			state.ForceFinal()
			break
		}
		tr := p.moves.Transitions[guess]
		parserinternals.Apply(state, tr.Move, tr.Label, p.labelHashes[guess])
	}

	// Step 4: write Token.Head / Token.Dep / Token.SentStart. After the
	// parser runs, every token's SentStart is known: 1 for sentence start,
	// 0 otherwise. Leaving "unknown" (-1) here would make Doc.Sents() panic
	// downstream on tokens the parser silently skipped.
	for i := range d.Tokens {
		d.Tokens[i].Head = i
		d.Tokens[i].Dep = p.rootLabel
		if state.IsSentStart(i) == 1 {
			d.Tokens[i].SentStart = 1
		} else {
			d.Tokens[i].SentStart = 0
		}
	}
	for _, arc := range state.Arcs() {
		d.Tokens[arc.Child].Head = arc.Head
		d.Tokens[arc.Child].Dep = arc.Label
	}

	parserinternals.Deprojectivize(d)
	return nil
}

// ScoreStateInternal computes the score vector for one state. Exported for
// the Task-9 unit test; production calls happen through Apply. The contract:
//
//   - ids: nF token indices (-1 = pad, routes to row 0 of cached)
//   - nF: feature count (8 for parser-state)
//   - cached: PrecomputableAffine output, flat ((T+1)*nF, nO*nP) — row r is
//     logical (token r/nF, feature r%nF); rows [0, nF) are the pad row.
//   - lowerBias: (nO, nP) flat — added per-state before maxout
//   - upperW: (nMoves, nO) flat — applied after maxout
//   - upperB: (nMoves,) — added to final scores
//   - nO, nP: hidden dims and maxout pieces
func ScoreStateInternal(scores []float32, ids []int32, nF int, cached, lowerBias, upperW, upperB []float32, nO, nP int) {
	stride := nO * nP
	unmaxed := make([]float32, stride)
	// sum_state_features: for each feature slot f, add cached[(id+1)*nF + f, :].
	for f := 0; f < nF; f++ {
		idx := int(ids[f])
		var rowBase int
		if idx < 0 {
			rowBase = f * stride // pad row at logical (0, f, :)
		} else {
			rowBase = ((idx + 1) * nF * stride) + (f * stride)
		}
		for j := 0; j < stride; j++ {
			unmaxed[j] += cached[rowBase+j]
		}
	}
	// Add bias.
	for j := 0; j < stride; j++ {
		unmaxed[j] += lowerBias[j]
	}
	// Maxout over nP for each o.
	hidden := make([]float32, nO)
	for o := 0; o < nO; o++ {
		best := unmaxed[o*nP]
		for p := 1; p < nP; p++ {
			if v := unmaxed[o*nP+p]; v > best {
				best = v
			}
		}
		hidden[o] = best
	}
	// Upper: scores[c] = Σ_o upperW[c*nO + o] * hidden[o] + upperB[c].
	nMoves := len(scores)
	for c := 0; c < nMoves; c++ {
		var s float32
		for o := 0; o < nO; o++ {
			s += upperW[c*nO+o] * hidden[o]
		}
		s += upperB[c]
		scores[c] = s
	}
}

// argMaxIfValid mirrors arg_max_if_valid (parser_model.pyx:227-233): return
// the index of the highest-scoring class whose is_valid bit is set, or -1.
func argMaxIfValid(scores []float32, isValid []int32) int {
	best := -1
	for i, v := range isValid {
		if v < 1 {
			continue
		}
		if best < 0 || scores[i] > scores[best] {
			best = i
		}
	}
	return best
}
