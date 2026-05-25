package gonum

import (
	"encoding/json"
	"testing"

	"github.com/bioshock/gospacy/v3/internal/diff"
	"github.com/stretchr/testify/require"
)

func TestHash_AgainstSample(t *testing.T) {
	fx, err := diff.LoadOpFixture(goldenSamplePath(t))
	require.NoError(t, err)
	c, ok := fx.Ops["hash"]
	require.True(t, ok)

	var idsArr struct {
		Shape []int    `json:"shape"`
		Dtype string   `json:"dtype"`
		Data  []uint64 `json:"data"`
	}
	require.NoError(t, json.Unmarshal(c.Inputs["ids"], &idsArr))

	var seed uint32
	require.NoError(t, json.Unmarshal(c.Inputs["seed"], &seed))

	want, err := c.Uint32Output()
	require.NoError(t, err)

	N := len(idsArr.Data)
	got := make([]uint32, N*4)
	Hash(got, idsArr.Data, seed)

	require.Equalf(t, want.Data, got, "hash mismatch for case=%q seed=%d", c.Name, seed)
}
