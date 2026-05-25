package vocab

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestVectors_EmptyLoadsCleanly(t *testing.T) {
	// en_core_web_sm has no word vectors but DOES have a vectors file and a
	// key2row map. The loader must accept both without erroring.
	dir := "../testdata/models/en_core_web_sm/vocab"
	if _, err := os.Stat(filepath.Join(dir, "vectors")); err != nil {
		t.Skip("model not downloaded")
	}
	vec, err := LoadVectorsDir(dir)
	require.NoError(t, err)
	require.Equal(t, 0, vec.NumKeys())
	require.Equal(t, 0, vec.Rows())
	// Cols may be zero or default (e.g. 300) depending on cfg; both fine.
}

func TestVectors_LookupOnEmpty(t *testing.T) {
	v := &Vectors{}
	row, ok := v.Row(123)
	require.False(t, ok)
	require.Nil(t, row)
}

// TestVectors_LoadMD_PopulatedLookup is Phase 7 Block C5's end-to-end check
// that LoadVectorsDir reads en_core_web_md's populated vector matrix +
// key2row index, and that Row(StringStore.Add(word)) resolves canonical
// English tokens to non-zero 300-dim vectors. Without this assertion the
// populated path is silently broken (the empty-case test covers sm but says
// nothing about the populated decode).
//
// SKIPped when md is not downloaded — the test does not block CI on the
// 55 MB asset, but it MUST pass on a fully bootstrapped machine.
func TestVectors_LoadMD_PopulatedLookup(t *testing.T) {
	mdPath := filepath.Join("..", "testdata", "models", "en_core_web_md", "vocab")
	if _, err := os.Stat(filepath.Join(mdPath, "vectors")); err != nil {
		t.Skipf("en_core_web_md not present: %v", err)
	}
	vec, err := LoadVectorsDir(mdPath)
	require.NoError(t, err)
	require.Equal(t, 300, vec.Cols(), "md vectors are 300-dimensional")
	require.Equal(t, 20000, vec.Rows(), "md ships with 20000 unique vector rows")
	require.Equal(t, 684830, vec.NumKeys(), "md key2row has 684830 keys (mapping to 20k unique rows)")

	// Resolve well-known English tokens through the StringStore hash and
	// verify they all land at valid row indices with non-zero vectors.
	store := NewStringStore()
	wantHits := []string{"apple", "the", "Google", "company", "running",
		"United", "Kingdom", "billion", "buying", "looking"}
	for _, w := range wantHits {
		h := store.Add(w)
		idx, ok := vec.RowIndex(h)
		require.Truef(t, ok, "expected %q (hash=%d) to be in md vectors", w, h)
		require.GreaterOrEqualf(t, idx, 0, "%q row idx", w)
		require.Lessf(t, idx, vec.Rows(), "%q row idx within bounds", w)
		row, hit := vec.Row(h)
		require.Truef(t, hit, "%q Row() lookup", w)
		require.Equal(t, 300, len(row))
		// A trained word vector has many non-zero components — a fresh
		// rand-init or zeroed row would fail this check.
		nonZero := 0
		for _, x := range row {
			if x != 0 {
				nonZero++
			}
		}
		require.Greaterf(t, nonZero, 200,
			"expected >200 non-zero entries in %q vector, got %d", w, nonZero)
	}

	// OOV check: a deliberately-bogus token should NOT be in the table.
	h := store.Add("asdfghjklqwerty12345")
	_, ok := vec.RowIndex(h)
	require.False(t, ok, "expected OOV token to miss")
	_, ok = vec.Row(h)
	require.False(t, ok)
}

// TestVectors_LoadLG_PopulatedLookup mirrors the md test on lg. lg has the
// same key2row entry count as md (684830) but a much larger underlying
// matrix (342918 unique rows vs md's 20000). Verifies the populated decode
// scales correctly to the larger matrix.
func TestVectors_LoadLG_PopulatedLookup(t *testing.T) {
	lgPath := filepath.Join("..", "testdata", "models", "en_core_web_lg", "vocab")
	if _, err := os.Stat(filepath.Join(lgPath, "vectors")); err != nil {
		t.Skipf("en_core_web_lg not present: %v", err)
	}
	vec, err := LoadVectorsDir(lgPath)
	require.NoError(t, err)
	require.Equal(t, 300, vec.Cols(), "lg vectors are 300-dimensional")
	require.Equal(t, 342918, vec.Rows(), "lg ships with 342918 unique vector rows")
	require.Equal(t, 684830, vec.NumKeys(), "lg key2row has 684830 keys")

	store := NewStringStore()
	for _, w := range []string{"apple", "Google", "the"} {
		h := store.Add(w)
		row, ok := vec.Row(h)
		require.Truef(t, ok, "expected %q to be in lg vectors", w)
		require.Equal(t, 300, len(row))
	}
}
