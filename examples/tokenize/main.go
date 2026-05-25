// Command tokenize is the smallest gospacy demo: build the English tokenizer
// rules, tokenize a string, print one row per token. No bundle, no neural pipes.
//
// Usage:
//
//	go run ./examples/tokenize [text]
//
// Default text is the sentence from the README quickstart.
package main

import (
	"fmt"
	"os"

	"github.com/bioshock/gospacy/v3/lang/en"
	"github.com/bioshock/gospacy/v3/tokenizer"
	"github.com/bioshock/gospacy/v3/vocab"
)

func main() {
	text := "Hello world. Don't go to the U.S.A. today!"
	if len(os.Args) >= 2 {
		text = os.Args[1]
	}

	rules, err := en.MakeRules()
	if err != nil {
		fmt.Fprintf(os.Stderr, "MakeRules: %v\n", err)
		os.Exit(1)
	}
	tk := tokenizer.New(rules)
	v := vocab.NewVocab()
	d := tk.ToDoc(v, text)

	fmt.Printf("Tokenize(%q) — %d tokens:\n", text, d.NumTokens())
	fmt.Printf("  %-4s  %-15s %-10s %-6s\n", "idx", "TEXT", "WHITESPACE", "SHAPE")
	for i := 0; i < d.NumTokens(); i++ {
		tok := d.Tokens[i]
		ws := tok.Whitespace
		if ws == "" {
			ws = "-"
		}
		fmt.Printf("  %-4d  %-15q %-10q %-6s\n", tok.Idx, tok.Text, ws, tok.Shape)
	}
}
