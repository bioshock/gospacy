package registry

import (
	"fmt"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
)

// FeatureExtractorColumns lists the attribute names accepted by
// spacy.FeatureExtractor.v1. These map to fields on doc.Token (resolved at
// runtime by buildFeatureExtractorV1's Forward).
var FeatureExtractorColumns = map[string]struct{}{
	"ORTH":     {},
	"NORM":     {},
	"LOWER":    {},
	"PREFIX":   {},
	"SUFFIX":   {},
	"SHAPE":    {},
	"SPACY":    {}, // 1 if the token is followed by whitespace, else 0
	"IS_SPACE": {}, // 1 if the token IS whitespace, else 0
	"LENGTH":   {}, // rune length
}

// buildFeatureExtractorV1 implements spacy.FeatureExtractor.v1.
//
// Phase 4 landed the structural factory (columns attr round-trip). Phase 4.5
// adds the forward pass: given a []*doc.Doc, produce []nn.Uint64s2d where
// row r column c is the hash of attribute `columns[c]` on Token r.
//
// Mirrors spacy.ml.featureextractor.FeatureExtractor.forward (which calls
// doc.to_array(columns) per doc). The supported column names are the 6 used
// by en_core_web_sm: NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE. Any other
// column name fails loud at Forward time — leaves caller code green for the
// real bundle without papering over unknown attrs.
func buildFeatureExtractorV1(cfg map[string]any) (*nn.Model, error) {
	cols, err := columnsFromCfg(cfg)
	if err != nil {
		return nil, err
	}
	return &nn.Model{
		Name:  "extract_features",
		Ops:   gonum.New(),
		Attrs: map[string]any{"columns": cols},
		Forward: func(m *nn.Model, X any) (any, error) {
			docs, ok := X.([]any)
			if !ok {
				return nil, fmt.Errorf("ExtractFeatures: expected []any (List[*doc.Doc]), got %T", X)
			}
			cols, _ := m.Attrs["columns"].([]string)
			out := make([]nn.Uint64s2d, len(docs))
			for i, raw := range docs {
				d, ok := raw.(*doc.Doc)
				if !ok {
					return nil, fmt.Errorf("ExtractFeatures: item %d is %T, want *doc.Doc", i, raw)
				}
				rows, ncols := d.NumTokens(), len(cols)
				data := make([]uint64, rows*ncols)
				for r := 0; r < rows; r++ {
					tok := d.Tokens[r]
					for c, name := range cols {
						v, err := tokenAttrHash(d, &tok, name)
						if err != nil {
							return nil, fmt.Errorf("ExtractFeatures doc %d tok %d: %w", i, r, err)
						}
						data[r*ncols+c] = v
					}
				}
				out[i] = nn.Uint64s2d{Data: data, Rows: rows, Cols: ncols}
			}
			return out, nil
		},
	}, nil
}

// tokenAttrHash dispatches column-name → uint64. Mirrors the relevant subset
// of spacy.attrs (only those columns en_core_web_sm uses). Unknown column
// names error so that loading a bundle with an unfamiliar attribute fails loud.
func tokenAttrHash(d *doc.Doc, tok *doc.Token, col string) (uint64, error) {
	switch col {
	case "NORM":
		return tok.Norm, nil
	case "PREFIX":
		return tok.Prefix, nil
	case "SUFFIX":
		return tok.Suffix, nil
	case "SHAPE":
		// Shape is a string field; intern via StringStore so it matches Python's
		// doc.to_array(SHAPE) which returns the StringStore hash.
		return d.Vocab.StringStore().Add(tok.Shape), nil
	case "SPACY":
		// SPACY = 1 iff trailing whitespace is non-empty.
		if tok.Whitespace != "" {
			return 1, nil
		}
		return 0, nil
	case "IS_SPACE":
		// IS_SPACE = 1 iff the surface text is entirely whitespace.
		for _, r := range tok.Text {
			if r != ' ' && r != '\t' && r != '\n' && r != '\r' {
				return 0, nil
			}
		}
		if tok.Text == "" {
			return 0, nil
		}
		return 1, nil
	default:
		return 0, fmt.Errorf("ExtractFeatures: unsupported column %q (supported: NORM,PREFIX,SUFFIX,SHAPE,SPACY,IS_SPACE)", col)
	}
}

func columnsFromCfg(cfg map[string]any) ([]string, error) {
	raw, ok := cfg["columns"]
	if !ok {
		return nil, fmt.Errorf("FeatureExtractor.v1: cfg missing 'columns'")
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("FeatureExtractor.v1: cfg['columns'] must be []any, got %T", raw)
	}
	out := make([]string, len(list))
	for i, v := range list {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("FeatureExtractor.v1: cfg['columns'][%d] is %T, want string", i, v)
		}
		out[i] = s
	}
	return out, nil
}
