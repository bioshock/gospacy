// Command bench measures gospacy.Bundle.Pipe throughput. Loads a bundle, runs
// 100 short sentences through the full pipeline (tokenize → tagger → parser
// → attribute_ruler → lemmatizer), and prints sentences/sec + tokens/sec.
//
// This is a per-run smoke test, not a microbenchmark. For per-op timings see
// nn/*_bench_test.go and BENCHMARKS.md.
//
// Usage:
//
//	go run ./examples/bench <path/to/en_core_web_sm>
package main

import (
	"fmt"
	"os"
	"time"

	"github.com/bioshock/gospacy/v3/bundle"
)

// sentences returns 100 short English sentences (4 base × 25 variants). Each
// has 5–9 tokens; the mix covers declaratives, questions, contractions,
// abbreviations.
func sentences() []string {
	bases := []string{
		"The cat sat on the mat.",
		"Don't go to the store today.",
		"Mr. Smith bought a new car.",
		"Where is the U.S. ambassador?",
	}
	out := make([]string, 0, 100)
	for i := 0; i < 25; i++ {
		for _, b := range bases {
			out = append(out, b)
		}
	}
	return out
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: bench <path/to/en_core_web_sm>")
		os.Exit(2)
	}
	bundlePath := os.Args[1]

	b, err := bundle.FromDisk(bundlePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FromDisk: %v\n", err)
		os.Exit(1)
	}

	corpus := sentences()

	// Warmup: one pass so the first call's sync.Once doesn't pollute timing.
	for _, s := range corpus {
		if _, err := b.Pipe(s); err != nil {
			fmt.Fprintf(os.Stderr, "Pipe (warmup) %q: %v\n", s, err)
			os.Exit(1)
		}
	}

	// Timed pass.
	var tokens int
	start := time.Now()
	for _, s := range corpus {
		d, err := b.Pipe(s)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Pipe %q: %v\n", s, err)
			os.Exit(1)
		}
		tokens += d.NumTokens()
	}
	elapsed := time.Since(start)

	sps := float64(len(corpus)) / elapsed.Seconds()
	tps := float64(tokens) / elapsed.Seconds()
	usPerTok := float64(elapsed.Microseconds()) / float64(tokens)

	fmt.Printf("Sentences:  %d\n", len(corpus))
	fmt.Printf("Tokens:     %d\n", tokens)
	fmt.Printf("Elapsed:    %v\n", elapsed)
	fmt.Printf("Throughput: %.1f sentences/sec, %.0f tokens/sec\n", sps, tps)
	fmt.Printf("Per token:  %.1f µs\n", usPerTok)
}
