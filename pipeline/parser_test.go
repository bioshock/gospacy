package pipeline_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/pipeline"
	"github.com/bioshock/gospacy/v3/vocab"
)

// TestScoreState_AddsCachedFeaturesPlusBiasAndMaxout walks a tiny synthetic
// (T+1=3, nF=2, nO=2, nP=2) cache and verifies the scorer (1) fetches
// cached[id+1] per feature, (2) routes id=-1 to row 0 = pad, (3) sums across
// nF features, (4) adds the (nO, nP) bias, (5) maxout-pools across nP,
// (6) applies upper.
func TestScoreState_AddsCachedFeaturesPlusBiasAndMaxout(t *testing.T) {
	// cached layout: (T+1)*nF rows × nO*nP cols. Row r corresponds to
	// (token r/nF, feature r%nF). Pad row is at rows [0..nF).
	nF, nO, nP := 2, 2, 2
	T := 2
	stride := nO * nP // 4
	cached := make([]float32, (T+1)*nF*stride)
	// Set token=1, f=0, (o=0,p=0)=1; f=1, (o=0,p=0)=10
	cached[(1*nF+0)*stride+0] = 1
	cached[(1*nF+1)*stride+0] = 10
	// Set token=2, f=0, (o=0,p=0)=2; f=1, (o=0,p=0)=20 (unused here)
	cached[(2*nF+0)*stride+0] = 2
	cached[(2*nF+1)*stride+0] = 20
	bias := []float32{0, 0, 0, 0} // (nO=2, nP=2)

	// ids[0]=0 → cached[(0+1)*nF + 0] = row 2 → cached[2*4+0]=1 for slot (o=0,p=0)
	// ids[1]=1 → cached[(1+1)*nF + 1] = row 5 → cached[5*4+0]=20 for slot (o=0,p=0)
	// unmaxed[0,0] = 1+20 = 21; other slots = 0.
	// hidden[0] = max(21, 0) = 21; hidden[1] = max(0,0) = 0.
	// upper W = [[1,0],[0,1]] (rows = c=0,c=1; cols = o)
	// scores[0] = 1*21+0*0 = 21; scores[1] = 0*21+1*0 = 0
	idsArr := [8]int32{}
	idsArr[0] = 0
	idsArr[1] = 1
	ids := idsArr[:]
	upperW := []float32{1, 0, 0, 1}
	upperB := []float32{0, 0}
	scores := make([]float32, 2)
	pipeline.ScoreStateInternal(scores, ids, nF, cached, bias, upperW, upperB, nO, nP)
	require.InDelta(t, float32(21), scores[0], 1e-5)
	require.InDelta(t, float32(0), scores[1], 1e-5)
}

// TestScoreState_PadRowOnMissingFeature confirms id=-1 routes to row f of
// the pad block (rows [0, nF)).
func TestScoreState_PadRowOnMissingFeature(t *testing.T) {
	nF, nO, nP := 2, 2, 2
	T := 1
	stride := nO * nP
	cached := make([]float32, (T+1)*nF*stride)
	// Pad row for f=0, (o=0,p=0) = 7
	cached[(0*nF+0)*stride+0] = 7
	// Pad row for f=1, (o=0,p=0) = 13
	cached[(0*nF+1)*stride+0] = 13
	bias := []float32{0, 0, 0, 0}
	idsArr := [8]int32{-1, -1}
	ids := idsArr[:]
	upperW := []float32{1, 0, 0, 1}
	upperB := []float32{0, 0}
	scores := make([]float32, 2)
	pipeline.ScoreStateInternal(scores, ids, nF, cached, bias, upperW, upperB, nO, nP)
	// unmaxed[0,0] = 7 + 13 = 20; hidden[0] = 20; scores[0] = 20.
	require.InDelta(t, float32(20), scores[0], 1e-5)
}

// TestParser_StubApplySetsRootsAndROOTLabel — Task-8 placeholder test: until
// Task 9 lands the real scorer, Parser.ApplyStub marks every token as root.
// (Replaced by the real differential test in Task 15.)
func TestParser_StubApplySetsRootsAndROOTLabel(t *testing.T) {
	v := vocab.NewVocab()
	rootHash := v.StringStore().Add("ROOT")
	p := &pipeline.Parser{}
	d := &doc.Doc{
		Vocab: v,
		Tokens: []doc.Token{
			{Text: "hi"},
			{Text: "there"},
			{Text: "."},
		},
	}
	require.NoError(t, p.ApplyStub(d, rootHash))
	for i, tok := range d.Tokens {
		require.Equalf(t, i, tok.Head, "tok %d head", i)
		require.Equalf(t, rootHash, tok.Dep, "tok %d dep", i)
	}
}
