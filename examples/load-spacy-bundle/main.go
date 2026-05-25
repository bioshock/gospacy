// Command load-spacy-bundle loads a .spacy model directory and prints:
//   - the bundle's language and pipeline,
//   - per-token table (text / tag / pos / morph / lemma / head / dep) via
//     Bundle.Pipe,
//   - per-pipe status footer (loaded / skipped — and why).
//
// Pipeline run: tokenize → tagger → parser → attribute_ruler → lemmatizer.
// NER and senter remain Skipped — see NOT_YET_PORTED.md.
//
// Usage:
//
//	load-spacy-bundle <path/to/.spacy/dir> [text]
package main

import (
	"fmt"
	"os"

	"github.com/bioshock/gospacy/v3/bundle"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: load-spacy-bundle <path> [text]")
		os.Exit(2)
	}
	path := os.Args[1]
	text := "Hello world. Don't go to the U.S.A. today!"
	if len(os.Args) >= 3 {
		text = os.Args[2]
	}

	b, err := bundle.FromDisk(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FromDisk: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Language: %s\n", b.Config.GetString("nlp.lang"))
	fmt.Printf("Pipeline: %v\n\n", b.Config.GetList("nlp.pipeline"))

	fmt.Printf("Pipe(%q):\n", text)
	d, err := b.Pipe(text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Pipe: %v\n", err)
		os.Exit(1)
	}
	ss := b.Vocab.StringStore()
	fmt.Printf("  %-4s %-15s %-6s %-6s %-25s %-12s %-4s %-8s\n",
		"IDX", "TEXT", "TAG", "POS", "MORPH", "LEMMA", "HEAD", "DEP")
	for i := 0; i < d.NumTokens(); i++ {
		tok := d.Tokens[i]
		tag, _ := ss.Lookup(tok.Tag)
		pos, _ := ss.Lookup(tok.POS)
		lemma, _ := ss.Lookup(tok.Lemma)
		dep, _ := ss.Lookup(tok.Dep)
		fmt.Printf("  %-4d %-15s %-6s %-6s %-25s %-12s %-4d %-8s\n",
			i, tok.Text, tag, pos, tok.Morph, lemma, tok.Head, dep)
	}
	fmt.Println()

	fmt.Println("Pipes (diagnostic):")
	for name, pipe := range b.Pipes {
		if pipe.Skipped {
			fmt.Printf("  %-20s SKIPPED  arch=%-40s reason=%s\n", name, pipe.Architecture, pipe.SkippedReason)
		} else {
			fmt.Printf("  %-20s loaded   arch=%s\n", name, pipe.Architecture)
		}
	}
}
