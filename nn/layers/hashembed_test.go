package layers

import (
	"testing"

	"github.com/bioshock/gospacy/v3/internal/murmur"
	"github.com/bioshock/gospacy/v3/nn"
	"github.com/bioshock/gospacy/v3/nn/backend/gonum"
	"github.com/stretchr/testify/require"
)

func TestHashEmbed_Forward(t *testing.T) {
	ops := gonum.New()
	nV, nO, seed := uint32(8), 3, uint32(0)
	m := HashEmbed(ops, nO, int(nV), seed)
	E := make([]float32, int(nV)*nO)
	for r := 0; r < int(nV); r++ {
		for j := 0; j < nO; j++ {
			E[r*nO+j] = float32(r)
		}
	}
	m.Params["E"] = E

	X := nn.Uint64s1d{Data: []uint64{42}}
	out, err := m.Predict(X)
	require.NoError(t, err)
	y := out.(nn.Floats2d)
	require.Equal(t, 1, y.Rows)
	require.Equal(t, nO, y.Cols)

	hashes := murmur.Hash3X86_128_Uint64(42, seed)
	var want [3]float32
	for k := 0; k < 4; k++ {
		row := int(hashes[k] % nV)
		for j := 0; j < nO; j++ {
			want[j] += E[row*nO+j]
		}
	}
	for j := 0; j < nO; j++ {
		require.InDelta(t, want[j], y.Data[j], 1e-6, "j=%d", j)
	}
}
