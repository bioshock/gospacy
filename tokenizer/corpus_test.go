package tokenizer_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/lang/en"
	"github.com/bioshock/gospacy/v3/tokenizer"
)

func TestTokenizer_TenKCorpus(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 10k corpus diff in -short mode")
	}
	rules, err := en.MakeRules()
	require.NoError(t, err)
	tk := tokenizer.New(rules)

	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "golden", "tokens-tokenizer-10k.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("10k golden missing: %v (run `make diff-test`)", err)
	}

	// Schema: {"corpus": ..., "sentences": [{"text": ..., "tokens": [{"orth": ..., "idx": ..., "ws": ...}]}]}
	// idx is a rune offset (Python str is unicode-indexed, so spaCy t.idx is already rune-based).
	var payload struct {
		Sentences []struct {
			Text   string `json:"text"`
			Tokens []struct {
				Orth string `json:"orth"`
				Idx  int    `json:"idx"`
				WS   bool   `json:"ws"`
			} `json:"tokens"`
		} `json:"sentences"`
	}
	require.NoError(t, json.Unmarshal(data, &payload))

	totalSentences := len(payload.Sentences)
	disagreeingSentences := 0
	var firstFailures []string

	for _, sent := range payload.Sentences {
		got := tk.Tokenize(sent.Text)
		agree := len(got) == len(sent.Tokens)
		if agree {
			for i, want := range sent.Tokens {
				if got[i].Orth != want.Orth || got[i].Idx != want.Idx || got[i].WS != want.WS {
					agree = false
					break
				}
			}
		}
		if !agree {
			disagreeingSentences++
			if len(firstFailures) < 5 {
				firstFailures = append(firstFailures, sent.Text)
			}
		}
	}

	agreement := 1.0 - float64(disagreeingSentences)/float64(totalSentences)
	t.Logf("tokenizer corpus: %d/%d sentences agree (%.4f%%)",
		totalSentences-disagreeingSentences, totalSentences, agreement*100)
	if disagreeingSentences > 0 {
		t.Logf("first %d disagreeing texts:", len(firstFailures))
		for _, s := range firstFailures {
			t.Logf("  %q", s)
		}
	}
	require.GreaterOrEqualf(t, agreement, 0.9999,
		"tokenizer agreement %.4f%% below 99.99%% target", agreement*100)
}
