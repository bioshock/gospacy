package tokenizer_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/bioshock/gospacy/v3/lang/en"
	"github.com/bioshock/gospacy/v3/tokenizer"
)

func TestTokenizer_GoldenCases(t *testing.T) {
	rules, err := en.MakeRules()
	if err != nil {
		t.Fatal(err)
	}
	tk := tokenizer.New(rules)

	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "golden", "tokenizer_cases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	var payload struct {
		Cases []struct {
			Text   string `json:"text"`
			Tokens []struct {
				Orth string `json:"orth"`
				Idx  int    `json:"idx"`
				WS   bool   `json:"ws"`
			} `json:"tokens"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatal(err)
	}

	for _, c := range payload.Cases {
		got := tk.Tokenize(c.Text)
		if len(got) != len(c.Tokens) {
			t.Errorf("tokenize(%q): got %d tokens, want %d\n  got:  %+v\n  want: %+v",
				c.Text, len(got), len(c.Tokens), got, c.Tokens)
			continue
		}
		for i, want := range c.Tokens {
			if got[i].Orth != want.Orth || got[i].Idx != want.Idx || got[i].WS != want.WS {
				t.Errorf("tokenize(%q) token %d: got {Orth:%q Idx:%d WS:%v} want {Orth:%q Idx:%d WS:%v}",
					c.Text, i, got[i].Orth, got[i].Idx, got[i].WS, want.Orth, want.Idx, want.WS)
			}
		}
	}
}
