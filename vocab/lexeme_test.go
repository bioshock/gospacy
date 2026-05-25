package vocab

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLexeme_AlphaWord(t *testing.T) {
	s := NewStringStore()
	lex := NewLexeme(s, "Hello")

	require.Equal(t, s.Add("Hello"), lex.Orth)
	require.Equal(t, s.Add("H"), lex.Prefix)   // default prefix len = 1 (matches spaCy)
	require.Equal(t, s.Add("llo"), lex.Suffix) // default suffix len = 3
	require.Equal(t, s.Add("Xxxxx"), lex.Shape)

	require.True(t, lex.IsAlpha)
	require.False(t, lex.IsDigit)
	require.False(t, lex.IsPunct)
	require.False(t, lex.IsSpace)
	require.False(t, lex.IsLower)
	require.False(t, lex.IsUpper)
	require.True(t, lex.IsTitle)
	require.True(t, lex.IsASCII)
}

func TestLexeme_Digits(t *testing.T) {
	s := NewStringStore()
	lex := NewLexeme(s, "1999")
	require.True(t, lex.IsDigit)
	require.False(t, lex.IsAlpha)
	require.True(t, lex.LikeNum)
	require.Equal(t, s.Add("dddd"), lex.Shape)
}

func TestLexeme_Punct(t *testing.T) {
	s := NewStringStore()
	lex := NewLexeme(s, ",")
	require.True(t, lex.IsPunct)
	require.False(t, lex.IsAlpha)
	require.False(t, lex.IsDigit)
}

func TestLexeme_ShapeShortWord(t *testing.T) {
	s := NewStringStore()
	lex := NewLexeme(s, "ab")
	require.Equal(t, s.Add("xx"), lex.Shape)
}

func TestLexeme_MatchesPythonGolden(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "golden", "lex_attrs.json")
	data, err := os.ReadFile(path)
	require.NoError(t, err)

	var payload struct {
		Words []struct {
			Text    string `json:"text"`
			Prefix  string `json:"prefix"`
			Suffix  string `json:"suffix"`
			Shape   string `json:"shape"`
			IsAlpha bool   `json:"is_alpha"`
			IsDigit bool   `json:"is_digit"`
			IsPunct bool   `json:"is_punct"`
			IsSpace bool   `json:"is_space"`
			IsLower bool   `json:"is_lower"`
			IsUpper bool   `json:"is_upper"`
			IsTitle bool   `json:"is_title"`
			IsASCII bool   `json:"is_ascii"`
		} `json:"words"`
	}
	require.NoError(t, json.Unmarshal(data, &payload))

	s := NewStringStore()
	for _, w := range payload.Words {
		lex := NewLexeme(s, w.Text)
		require.Equalf(t, s.Add(w.Prefix), lex.Prefix, "%q.prefix", w.Text)
		require.Equalf(t, s.Add(w.Suffix), lex.Suffix, "%q.suffix", w.Text)
		require.Equalf(t, s.Add(w.Shape), lex.Shape, "%q.shape", w.Text)
		require.Equalf(t, w.IsAlpha, lex.IsAlpha, "%q.is_alpha", w.Text)
		require.Equalf(t, w.IsDigit, lex.IsDigit, "%q.is_digit", w.Text)
		require.Equalf(t, w.IsPunct, lex.IsPunct, "%q.is_punct", w.Text)
		require.Equalf(t, w.IsSpace, lex.IsSpace, "%q.is_space", w.Text)
		require.Equalf(t, w.IsLower, lex.IsLower, "%q.is_lower", w.Text)
		require.Equalf(t, w.IsUpper, lex.IsUpper, "%q.is_upper", w.Text)
		require.Equalf(t, w.IsTitle, lex.IsTitle, "%q.is_title", w.Text)
		require.Equalf(t, w.IsASCII, lex.IsASCII, "%q.is_ascii", w.Text)
	}
}
