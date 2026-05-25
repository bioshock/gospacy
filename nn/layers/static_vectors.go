package layers

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/vocab"
)

// StaticVectors mirrors spacy.ml.staticvectors.StaticVectors.
//
// Inference contract (two dispatch arms, matching upstream's polymorphism on
// input kind):
//
//  1. nn.Ints1d of resolved row indices → nn.Floats2d{Rows, nO}. Used by
//     per-layer unit tests where the lookup is mocked. Row index -1 marks an
//     OOV token (zero output row).
//
//  2. []any (List[*doc.Doc]) → nn.Ragged{Cols=nO, Lengths=per-doc-tok-count}.
//     Used in the real tok2vec graph for en_core_web_md / _lg. The layer
//     resolves per-token ORTH hashes through Attrs["vocab_vectors"]
//     (*vocab.Vectors) to row indices, gathers the rows, applies the
//     projection, and zero-outs OOV rows post-projection.
//
// Dims:
//
//	nO : output dim (= MultiHashEmbed.width, e.g. 96 in en_core_web_md/lg)
//	nM : vector-table column width (= upstream "vecW", typically 300)
//
// Params:
//
//	W : (nO, nM) linear projection (= spacy's `model.get_param("W")`)
//
// Attrs (populated externally — NOT serialised in the thinc payload):
//
//	"vocab_vectors" : *vocab.Vectors    (bundle loader sets this from
//	                                     vocab/vectors after FromBytes)
//	"vectors"       : []float32         (fallback flat-table for unit tests
//	                                     that skip the Vocab dependency)
//	"nV"            : int               (row count when "vectors" is set)
//
// Note: in production (md/lg), the layer reads from "vocab_vectors". In unit
// tests we set "vectors"+"nV" directly so we don't need a *vocab.Vectors.
func StaticVectors(ops nn.Ops, nO, nM int) *nn.Model {
	return &nn.Model{
		Name:   "static_vectors",
		Ops:    ops,
		Dims:   map[string]int{"nO": nO, "nM": nM},
		Params: map[string][]float32{"W": nil},
		Attrs: map[string]any{
			"vocab_vectors": (*vocab.Vectors)(nil),
			"vectors":       []float32(nil),
			"nV":            int(0),
		},
		Forward: func(m *nn.Model, X any) (any, error) {
			switch in := X.(type) {
			case nn.Ints1d:
				out, err := projectRows(m, in.Data)
				if err != nil {
					return nil, err
				}
				return out, nil
			case []any:
				return forwardDocs(m, in)
			default:
				return nil, fmt.Errorf("StaticVectors: expected Ints1d or []any, got %T", X)
			}
		},
	}
}

// projectRows gathers vector rows (per the int32 row index slice; -1 → zero
// row) and applies the W.T projection. Returns Floats2d (len(rows), nO).
//
// Reads the vector table from Attrs["vectors"] (raw flat slice + nV) when set,
// else from Attrs["vocab_vectors"].
func projectRows(m *nn.Model, rows []int32) (nn.Floats2d, error) {
	nO := m.Dims["nO"]
	nM := m.Dims["nM"]
	W := m.Params["W"]
	if len(W) != nO*nM {
		return nn.Floats2d{}, fmt.Errorf(
			"StaticVectors: W has %d floats, expected nO*nM=%d (nO=%d nM=%d)",
			len(W), nO*nM, nO, nM,
		)
	}
	N := len(rows)
	if N == 0 {
		return nn.Floats2d{Data: nil, Rows: 0, Cols: nO}, nil
	}

	flatVecs, nV, err := tableFrom(m, nM)
	if err != nil {
		return nn.Floats2d{}, err
	}

	lookup := make([]float32, N*nM)
	for i, r := range rows {
		ri := int(r)
		if ri < 0 {
			continue // OOV → leave zeros
		}
		if ri >= nV {
			return nn.Floats2d{}, fmt.Errorf("StaticVectors: row index %d out of range [0, %d)", ri, nV)
		}
		copy(lookup[i*nM:(i+1)*nM], flatVecs[ri*nM:(ri+1)*nM])
	}

	out := make([]float32, N*nO)
	zeroBias := make([]float32, nO)
	m.Ops.Affine(out, lookup, N, nM, W, nO, zeroBias)
	return nn.Floats2d{Data: out, Rows: N, Cols: nO}, nil
}

// tableFrom returns the underlying vector table as a flat float32 slice and
// nV. Prefers Attrs["vocab_vectors"] (production path); falls back to
// Attrs["vectors"]+Attrs["nV"] (unit-test path).
func tableFrom(m *nn.Model, nM int) ([]float32, int, error) {
	if vec, ok := m.Attrs["vocab_vectors"].(*vocab.Vectors); ok && vec != nil && vec.Rows() > 0 {
		if vec.Cols() != nM {
			return nil, 0, fmt.Errorf(
				"StaticVectors: vocab.Vectors cols=%d, expected nM=%d",
				vec.Cols(), nM,
			)
		}
		return vec.Data(), vec.Rows(), nil
	}
	flatVecs, _ := m.Attrs["vectors"].([]float32)
	nV, _ := m.Attrs["nV"].(int)
	if nV > 0 && len(flatVecs) != nV*nM {
		return nil, 0, fmt.Errorf(
			"StaticVectors: vectors has %d floats, expected nV*nM=%d (nV=%d nM=%d)",
			len(flatVecs), nV*nM, nV, nM,
		)
	}
	return flatVecs, nV, nil
}

// forwardDocs resolves per-token ORTH hashes through vocab.Vectors, applies
// the projection, and emits a Ragged matching the per-doc token counts.
// Mirrors upstream's spacy.ml.staticvectors.forward:
//
//	keys = flatten(doc.to_array(ORTH) for doc in docs)
//	rows = vocab.vectors.find(keys=keys)
//	V = vocab.vectors.data[rows]; vectors_data = gemm(V, W, trans2=True)
//	vectors_data[rows < 0] = 0
//	return Ragged(vectors_data, lengths=[len(doc) for doc in docs])
func forwardDocs(m *nn.Model, docs []any) (nn.Ragged, error) {
	nO := m.Dims["nO"]
	totalTokens := 0
	for i, raw := range docs {
		d, ok := raw.(*doc.Doc)
		if !ok {
			return nn.Ragged{}, fmt.Errorf("StaticVectors: item %d is %T, want *doc.Doc", i, raw)
		}
		totalTokens += d.NumTokens()
	}
	if totalTokens == 0 {
		return nn.Ragged{Data: nil, Lengths: make([]int32, len(docs)), Cols: nO}, nil
	}

	vec, ok := m.Attrs["vocab_vectors"].(*vocab.Vectors)
	if !ok || vec == nil {
		return nn.Ragged{}, fmt.Errorf("StaticVectors: missing vocab_vectors attr for doc-mode forward")
	}

	rows := make([]int32, 0, totalTokens)
	lengths := make([]int32, len(docs))
	for i, raw := range docs {
		d := raw.(*doc.Doc) // type-checked above
		lengths[i] = int32(d.NumTokens())
		for _, tok := range d.Tokens {
			// upstream uses ORTH (= the literal surface form hash) as the
			// lookup key. Token.Orth is precisely that — see
			// tokenizer/tokenizer.go which interns Tok.Text via
			// vocab.StringStore.Add.
			r, hit := vec.Row(tok.Orth)
			if !hit {
				rows = append(rows, -1)
				continue
			}
			// vec.Row returns the row slice; recover the row index from the
			// underlying map by re-querying through a row-index-by-hash
			// lookup. We expose this via Vectors.RowIndex() (added below).
			_ = r
			ri, _ := vec.RowIndex(tok.Orth)
			rows = append(rows, int32(ri))
		}
	}

	proj, err := projectRows(m, rows)
	if err != nil {
		return nn.Ragged{}, err
	}
	return nn.Ragged{Data: proj.Data, Lengths: lengths, Cols: nO}, nil
}
