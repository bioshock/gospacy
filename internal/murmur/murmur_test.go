package murmur

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

type murmurFixture struct {
	Hash64 []struct {
		Key      string `json:"key"`
		Seed     uint32 `json:"seed"`
		ValueDec string `json:"value_dec"`
	} `json:"hash64"`
	Hash3X86128Uint64 []struct {
		Key   uint64    `json:"key"`
		Seed  uint32    `json:"seed"`
		Value [4]uint32 `json:"value"`
	} `json:"hash3_x86_128_uint64"`
}

func loadFixture(t *testing.T) *murmurFixture {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..")
	path := filepath.Join(root, "testdata", "golden", "murmur_vectors.json")
	b, err := os.ReadFile(path)
	require.NoError(t, err, "run `testharness/.venv/bin/python testharness/dump_murmur.py` first")
	var fx murmurFixture
	require.NoError(t, json.Unmarshal(b, &fx))
	return &fx
}

func TestHash64A_AgainstPython(t *testing.T) {
	fx := loadFixture(t)
	require.NotEmpty(t, fx.Hash64)
	for _, c := range fx.Hash64 {
		want, err := strconv.ParseUint(c.ValueDec, 10, 64)
		require.NoError(t, err)
		got := Hash64A([]byte(c.Key), c.Seed)
		require.Equalf(t, want, got, "key=%q seed=%d", c.Key, c.Seed)
	}
}

func TestHash3X86_128_Uint64_AgainstPython(t *testing.T) {
	fx := loadFixture(t)
	require.NotEmpty(t, fx.Hash3X86128Uint64)
	for _, c := range fx.Hash3X86128Uint64 {
		got := Hash3X86_128_Uint64(c.Key, c.Seed)
		require.Equalf(t, c.Value, got, "key=%d seed=%d", c.Key, c.Seed)
	}
}
