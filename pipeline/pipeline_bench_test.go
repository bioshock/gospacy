package pipeline_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/bioshock/gospacy/v3/bundle"
)

// BenchmarkBundle_Pipe_FixtureSentences benchmarks Bundle.Pipe across the 8
// canonical pipeline_cases.json sentences. Block B uses this to identify the
// per-record cost of the full pipeline (tokenize + tok2vec + tagger + parser
// + attribute_ruler + lemmatizer) on short text.
func BenchmarkBundle_Pipe_FixtureSentences(b *testing.B) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		b.Skipf("en_core_web_sm not present: %s", bundlePath)
	}
	rawCases, err := os.ReadFile(filepath.Join("..", "testharness", "pipeline_cases.json"))
	if err != nil {
		b.Fatalf("read cases: %v", err)
	}
	var casesFile struct {
		Cases []struct {
			ID   string `json:"id"`
			Text string `json:"text"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(rawCases, &casesFile); err != nil {
		b.Fatalf("parse cases: %v", err)
	}
	bd, err := bundle.FromDisk(bundlePath)
	if err != nil {
		b.Fatalf("FromDisk: %v", err)
	}
	// Warmup: ensure Bundle.ensureComponents has run.
	if _, err := bd.Pipe(casesFile.Cases[0].Text); err != nil {
		b.Fatalf("warmup: %v", err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c := casesFile.Cases[i%len(casesFile.Cases)]
		if _, err := bd.Pipe(c.Text); err != nil {
			b.Fatalf("Pipe(%q): %v", c.Text, err)
		}
	}
}

// BenchmarkBundle_Pipe_LongClaimStyle benchmarks Bundle.Pipe on a single
// long claim-style sentence (comma- and semicolon-heavy trademark-class
// description) — the shape of input where the ~2× latency gap (16.2 ms
// Python vs 31.6 ms Go per record) showed up early in the port. Kept as
// a regression bench so per-Pipe latency on this workload stays tracked.
func BenchmarkBundle_Pipe_LongClaimStyle(b *testing.B) {
	bundlePath := filepath.Join("..", "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		b.Skipf("en_core_web_sm not present: %s", bundlePath)
	}
	const text = "Computer software; application software; downloadable software for trading crypto-products and providing crypto-currency information; authentication and authorization software; automatic banking machines."
	bd, err := bundle.FromDisk(bundlePath)
	if err != nil {
		b.Fatalf("FromDisk: %v", err)
	}
	if _, err := bd.Pipe(text); err != nil {
		b.Fatalf("warmup: %v", err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := bd.Pipe(text); err != nil {
			b.Fatalf("Pipe: %v", err)
		}
	}
}
