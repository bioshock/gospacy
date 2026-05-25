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

// NER is the EntityRecognizer pipe. Sibling of Parser: same scorer shape
// (PrecomputableAffine lower + Linear upper), same greedy decode loop, but
// the TransitionSystem is BiluoPushDown (not ArcEager) and the writeback
// fills Token.EntIOB + Token.EntType (not Head/Dep).
//
// Inference only — greedy decoding (beam_width=1). Beam search, oracle, and
// learn_tokens are out of scope; see NOT_YET_PORTED.md.
type NER struct {
	moves       *parserinternals.TransitionSystem
	labelHashes []uint64 // per-class StringStore hash of moves.Transitions[i].Label (0 for "")
	model       *nn.Model
	cfg         nerCfg
	src         BundleSource
}

type nerCfg struct {
	BeamWidth         int     `json:"beam_width"`
	BeamDensity       float64 `json:"beam_density"`
	LearnTokens       bool    `json:"learn_tokens"`
	MinActionFreq     int     `json:"min_action_freq"`
	IncorrectSpansKey *string `json:"incorrect_spans_key"`
}

// NewNER loads moves + cfg from <bundle>/ner/{moves,cfg}, interns the move
// labels into the bundle Vocab, and binds the model. Returns an error when
// beam_width > 1 (beam search is not supported — see NOT_YET_PORTED.md).
func NewNER(b BundleSource) (*NER, error) {
	model, skipped, skipReason, ok := b.PipeLookup("ner")
	if !ok {
		return nil, fmt.Errorf("NER: bundle has no 'ner' pipe")
	}
	if skipped || model == nil {
		return nil, fmt.Errorf("NER: pipe skipped (%s)", skipReason)
	}

	cfgPath := filepath.Join(b.BundlePath(), "ner", "cfg")
	cfgBytes, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, fmt.Errorf("NER: read ner/cfg: %w", err)
	}
	var cfg nerCfg
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return nil, fmt.Errorf("NER: parse ner/cfg: %w", err)
	}
	if cfg.BeamWidth > 1 {
		return nil, fmt.Errorf("NER: beam_width=%d not supported (greedy decode only)", cfg.BeamWidth)
	}
	if cfg.LearnTokens {
		return nil, fmt.Errorf("NER: learn_tokens=true not supported (inference only)")
	}

	movesPath := filepath.Join(b.BundlePath(), "ner", "moves")
	ts, err := parserinternals.LoadBiluoMoves(movesPath)
	if err != nil {
		return nil, fmt.Errorf("NER: load moves: %w", err)
	}

	// Intern labels and confirm the upper Linear's output dimension matches
	// our LoadBiluoMoves count. The on-disk model's nO is set by FromBytes;
	// if the two diverge, the scoring loop would read out-of-bounds.
	ss := b.BundleVocab().StringStore()
	labelHashes := make([]uint64, ts.NMoves)
	for i, tr := range ts.Transitions {
		if tr.Label == "" {
			labelHashes[i] = 0
		} else {
			labelHashes[i] = ss.Add(tr.Label)
		}
	}
	if upper := model.Refs["upper"]; upper != nil {
		if nMoves := upper.Dims["nO"]; nMoves != ts.NMoves {
			return nil, fmt.Errorf("NER: upper.nO=%d != moves.NMoves=%d", nMoves, ts.NMoves)
		}
	}

	return &NER{
		moves:       ts,
		labelHashes: labelHashes,
		model:       model,
		cfg:         cfg,
		src:         b,
	}, nil
}

// ApplyStub is a degenerate Apply used by unit tests that don't want to run
// the full scorer: tags every token as O with EntType 0. Production callers
// use Apply.
func (n *NER) ApplyStub(d *doc.Doc) error {
	for i := range d.Tokens {
		d.Tokens[i].EntIOB = 2 // O
		d.Tokens[i].EntType = 0
	}
	return nil
}

// Apply runs a greedy BILUO decode over d, given the NER tok2vec output
// (T rows × 64 cols, already projected through the NER pipe's
// tok2vec_chain). The NER pipe has its own non-listener Tok2Vec.v2 — the
// bundle invokes that chain separately and passes the projected
// Floats2d directly. Writes Token.EntIOB and Token.EntType.
//
// The State is seeded with the Doc's SentStart bits (set by the parser
// before NER runs) so BiluoIsValid's "no entity across sentence
// boundaries" gate works. Without this seed, NER would happily span
// sentence breaks.
//
// On a decode failure (argMaxIfValid returns -1), the loop forces the
// state final and fills any unwritten EntIOB slot with O — matching
// upstream's set_annotations behaviour for tokens left at ent_iob == 0.
func (n *NER) Apply(d *doc.Doc, nerTok2VecProjected nn.Floats2d) error {
	if d.NumTokens() == 0 {
		return nil
	}
	T := d.NumTokens()
	if nerTok2VecProjected.Rows != T {
		return fmt.Errorf("NER.Apply: tok2vec rows %d != tokens %d", nerTok2VecProjected.Rows, T)
	}

	// Step 1: precompute lower (PrecomputableAffine) on the projected
	// tok2vec output. Same path as Parser.Apply Step 2.
	lower := n.model.Refs["lower"]
	if lower == nil {
		return fmt.Errorf("NER.Apply: model.Refs[\"lower\"] is nil")
	}
	lowerRaw, err := lower.Predict(nerTok2VecProjected)
	if err != nil {
		return fmt.Errorf("NER.Apply: lower: %w", err)
	}
	lowerOut, ok := lowerRaw.(nn.Floats2d)
	if !ok {
		return fmt.Errorf("NER.Apply: lower produced %T, want Floats2d", lowerRaw)
	}
	nF := lower.Dims["nF"]
	nO := lower.Dims["nO"]
	nP := lower.Dims["nP"]
	// NER uses a 3-feature template (B(0), E(0), B(0)-1) — see _state.pxd
	// set_context_tokens n==3 branch. Parser uses nF=8.
	if nF != 3 {
		return fmt.Errorf("NER.Apply: nF=%d, expected 3 (NER feature template)", nF)
	}
	if lowerOut.Rows != (T+1)*nF || lowerOut.Cols != nO*nP {
		return fmt.Errorf("NER.Apply: lower output shape mismatch (rows=%d cols=%d, want %d×%d)",
			lowerOut.Rows, lowerOut.Cols, (T+1)*nF, nO*nP)
	}

	upper := n.model.Refs["upper"]
	if upper == nil {
		return fmt.Errorf("NER.Apply: model.Refs[\"upper\"] is nil")
	}
	upperW := upper.Params["W"]
	upperB := upper.Params["b"]
	lowerBias := lower.Params["b"]
	nMoves := upper.Dims["nO"]
	if nMoves != n.moves.NMoves {
		return fmt.Errorf("NER.Apply: upper.nO=%d != moves.NMoves=%d", nMoves, n.moves.NMoves)
	}

	// Step 2: build the State and seed sent_starts from Doc.Tokens. The
	// parser writes SentStart before NER runs; BiluoIsValid reads it to
	// gate B/I across sentence boundaries.
	state := parserinternals.NewState(T)
	for i := range d.Tokens {
		if d.Tokens[i].SentStart == 1 {
			state.SetSentStart(i, 1)
		}
	}

	// Step 3: greedy BILUO decode.
	isValid := make([]int32, n.moves.NMoves)
	scores := make([]float32, n.moves.NMoves)
	var idsArr [3]int32
	ids := idsArr[:]
	for !state.IsFinal() {
		state.SetContextTokens3NER(&idsArr)
		ScoreStateInternal(scores, ids, nF, lowerOut.Data, lowerBias, upperW, upperB, nO, nP)
		parserinternals.BiluoSetValid(isValid, state, n.moves, n.labelHashes)
		guess := argMaxIfValid(scores, isValid)
		if guess < 0 {
			// Force final: any unmarked token gets O (matches spaCy's
			// set_annotations loop that flips ent_iob==0 → 2).
			for i := 0; i < T; i++ {
				if state.EntIOB(i) == 0 {
					state.SetEntIOB(i, 2)
				}
			}
			state.ForceFinal()
			break
		}
		tr := n.moves.Transitions[guess]
		parserinternals.BiluoApply(state, tr.Move, tr.Label, n.labelHashes[guess])
	}

	// Step 4: writeback. Token.EntIOB stores the per-token BILUO code
	// (0 missing, 1 I-, 2 O, 3 B-). Token.EntType is the label hash of
	// the containing entity (0 for OUT tokens).
	for i := range d.Tokens {
		d.Tokens[i].EntIOB = state.EntIOB(i)
		d.Tokens[i].EntType = 0
	}
	for _, ent := range state.Entities() {
		for i := ent.Start; i < ent.End; i++ {
			if i >= 0 && i < T {
				d.Tokens[i].EntType = ent.Label
			}
		}
	}
	return nil
}
