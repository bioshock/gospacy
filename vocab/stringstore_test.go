package vocab

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestStringStore_EmptyIsZero(t *testing.T) {
	s := NewStringStore()
	h, ok := s.Get("")
	require.True(t, ok)
	require.Equal(t, uint64(0), h)
	got, ok := s.Lookup(0)
	require.True(t, ok)
	require.Equal(t, "", got)
}

func TestStringStore_AddRoundTrip(t *testing.T) {
	s := NewStringStore()
	h := s.Add("hello")
	require.NotEqual(t, uint64(0), h)
	got, ok := s.Lookup(h)
	require.True(t, ok)
	require.Equal(t, "hello", got)
	h2, ok := s.Get("hello")
	require.True(t, ok)
	require.Equal(t, h, h2)
}

func TestStringStore_AddIdempotent(t *testing.T) {
	s := NewStringStore()
	h1 := s.Add("x")
	h2 := s.Add("x")
	require.Equal(t, h1, h2)
	require.Equal(t, 1, s.Len()) // empty string + "x" - 1 (empty doesn't count for Len)
}

func TestStringStore_MatchesPythonGolden(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "golden", "stringstore.json")
	data, err := os.ReadFile(path)
	require.NoError(t, err)

	var payload struct {
		Strings []struct {
			Text string `json:"text"`
			Hash string `json:"hash"`
		} `json:"strings"`
	}
	require.NoError(t, json.Unmarshal(data, &payload))

	s := NewStringStore()
	for _, c := range payload.Strings {
		want, err := strconv.ParseUint(c.Hash, 10, 64)
		require.NoError(t, err)
		got := s.Add(c.Text)
		require.Equalf(t, want, got, "hash(%q): want %d got %d", c.Text, want, got)
	}
}

func TestStringStore_LoadJSON(t *testing.T) {
	s := NewStringStore()
	require.NoError(t, s.LoadJSON([]byte(`["alpha","beta","gamma"]`)))
	for _, w := range []string{"alpha", "beta", "gamma"} {
		_, ok := s.Get(w)
		require.Truef(t, ok, "expected %q interned", w)
	}
	require.Equal(t, 3, s.Len())
}

func TestStringStore_LoadJSON_RejectsNonArray(t *testing.T) {
	s := NewStringStore()
	err := s.LoadJSON([]byte(`{"strings":["alpha"]}`))
	require.Error(t, err)
	require.Contains(t, err.Error(), "StringStore.LoadJSON")
}
