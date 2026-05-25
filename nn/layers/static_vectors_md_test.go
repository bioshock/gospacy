package layers

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
)

// TestStaticVectors_MDGoldenParity is Phase 7 Block C3's per-layer parity
// check: for ~10 known English tokens, gospacy's StaticVectors Forward output
// must agree with spaCy's exact gemm path to within 1e-6 (single-precision
// rounding).
//
// Why this matters: if W is loaded wrong (transposed, sliced incorrectly), or
// if our Affine differs from spaCy's gemm in row-major vs col-major
// interpretation, the downstream tok2vec output drifts and the tagger/parser
// strict-100% goldens fail by tens of tokens. Pinning numerical parity at the
// per-layer boundary makes that class of bug a one-line diagnosis instead of
// a 68-token tagger-test diff.
//
// Inputs:
//   - testdata/golden/static_vectors_md.json
//     · nO, nM, per-sample {row, vec, expected out}
//   - testdata/golden/static_vectors_W_md.bin
//     · raw little-endian float32 W flattened (nO*nM*4 bytes)
//
// Both are regenerated via
// `GOSPACY_MODEL=en_core_web_md testharness/.venv/bin/python testharness/dump_static_vectors.py`.
//
// Skipped if the goldens are not present (e.g. md not downloaded yet).
func TestStaticVectors_MDGoldenParity(t *testing.T) {
	goldenJSON := filepath.Join("..", "..", "testdata", "golden", "static_vectors_md.json")
	goldenW := filepath.Join("..", "..", "testdata", "golden", "static_vectors_W_md.bin")
	if _, err := os.Stat(goldenJSON); err != nil {
		t.Skipf("static_vectors_md.json not present: %v", err)
	}
	rawJSON, err := os.ReadFile(goldenJSON)
	require.NoError(t, err)
	type sample struct {
		Key     string    `json:"key"`
		KeyHash uint64    `json:"key_hash"`
		Row     int32     `json:"row"`
		Vec     []float32 `json:"vec"`
		Out     []float32 `json:"out"`
	}
	var payload struct {
		NO      int      `json:"nO"`
		NM      int      `json:"nM"`
		Samples []sample `json:"samples"`
	}
	require.NoError(t, json.Unmarshal(rawJSON, &payload))
	require.Greater(t, len(payload.Samples), 0)
	require.Equal(t, 96, payload.NO)
	require.Equal(t, 300, payload.NM)

	rawW, err := os.ReadFile(goldenW)
	require.NoError(t, err)
	expectedWBytes := payload.NO * payload.NM * 4
	require.Equalf(t, expectedWBytes, len(rawW),
		"W bin file has %d bytes, expected nO*nM*4=%d", len(rawW), expectedWBytes)
	W := make([]float32, payload.NO*payload.NM)
	for i := range W {
		bits := binary.LittleEndian.Uint32(rawW[i*4:])
		W[i] = math.Float32frombits(bits)
	}

	// Build the StaticVectors layer with the golden W. Stitch a single
	// row-vector table containing all sample rows, plus a synthetic OOV row
	// at index 0. We use the per-sample `vec` (already gathered from
	// vocab.vectors) so the lookup step is mocked — what we're testing here
	// is W.T @ vec equivalence, NOT the find()/key2row resolution (that's
	// covered by C5 in vocab/vectors_test.go).
	ops := gonum.New()
	model := StaticVectors(ops, payload.NO, payload.NM)
	model.Params["W"] = W

	// Build a synthetic vectors table: one row per in-vocab sample, in the
	// order they appear in `samples`. OOV samples (Row=-1) feed -1 into the
	// layer to exercise the zero-row branch.
	flat := []float32{}
	indices := []int32{}
	for _, s := range payload.Samples {
		if s.Row < 0 {
			indices = append(indices, -1)
			continue
		}
		require.Equalf(t, payload.NM, len(s.Vec),
			"sample %q: vec has %d floats, expected nM=%d", s.Key, len(s.Vec), payload.NM)
		row := int32(len(flat) / payload.NM)
		flat = append(flat, s.Vec...)
		indices = append(indices, row)
	}
	model.Attrs["vectors"] = flat
	model.Attrs["nV"] = len(flat) / payload.NM

	out, err := model.Predict(nn.Ints1d{Data: indices})
	require.NoError(t, err)
	got, ok := out.(nn.Floats2d)
	require.True(t, ok)
	require.Equal(t, len(indices), got.Rows)
	require.Equal(t, payload.NO, got.Cols)

	maxAbsDiff := float32(0)
	for i, s := range payload.Samples {
		rowOut := got.Data[i*payload.NO : (i+1)*payload.NO]
		// OOV row: expect all zeros and golden Out should be all zeros.
		if s.Row < 0 {
			for j, v := range rowOut {
				require.InDeltaf(t, 0.0, v, 1e-6,
					"sample %q (OOV) col %d: got %f, expected 0", s.Key, j, v)
				if v < 0 {
					if -v > maxAbsDiff {
						maxAbsDiff = -v
					}
				} else if v > maxAbsDiff {
					maxAbsDiff = v
				}
			}
			continue
		}
		require.Equalf(t, payload.NO, len(s.Out),
			"sample %q: golden Out has %d floats, expected nO=%d", s.Key, len(s.Out), payload.NO)
		for j, want := range s.Out {
			got := rowOut[j]
			diff := got - want
			if diff < 0 {
				diff = -diff
			}
			if diff > maxAbsDiff {
				maxAbsDiff = diff
			}
			require.InDeltaf(t, want, got, 1e-5,
				"sample %q col %d: want %f got %f (diff=%g)", s.Key, j, want, got, diff)
		}
	}
	t.Logf("StaticVectors md parity: max-abs-diff = %g across %d samples × %d dims",
		maxAbsDiff, len(payload.Samples), payload.NO)
}
