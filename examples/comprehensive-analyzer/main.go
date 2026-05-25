// Command comprehensive-analyzer is a Go port of a comprehensive spaCy
// usage demo. It exercises the full inference path on en_core_web_md:
// tokenization + lemmatization + POS/Tag + IS_ALPHA / IS_STOP, named
// entities with human-readable label explanations, noun chunks,
// dependency parsing, sentence segmentation, a hand-rolled equivalent
// of spacy.matcher.Matcher (the generic Matcher is NOT_YET_PORTED in
// gospacy — see NOT_YET_PORTED.md), and document similarity via
// mean-pooled vectors (spaCy's default doc.similarity behaviour).
//
// Usage:
//
//	comprehensive-analyzer <path/to/en_core_web_md>
//
// Use en_core_web_md or _lg — _sm has no vectors and the similarity
// section will report 0. Get a bundle via:
//
//	testharness/.venv/bin/python -m spacy download en_core_web_md
//	# then copy it into testdata/models/ per testharness/download_assets.sh
package main

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/bioshock/gospacy/v3/bundle"
	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/internal/lexflags"
	"github.com/bioshock/gospacy/v3/matcher"
	"github.com/bioshock/gospacy/v3/vocab"
)

// analyzer wraps a loaded gospacy bundle. Mirrors the
// ComprehensiveSpacyAnalyzer Python class — single owner per goroutine
// (Bundle.Pipe is single-goroutine by contract; use bundle.Clone for
// parallel fan-out).
type analyzer struct {
	b *bundle.Bundle
}

func newAnalyzer(modelPath string) (*analyzer, error) {
	fmt.Printf("Loading spaCy model from: %s\n", modelPath)
	b, err := bundle.FromDisk(modelPath)
	if err != nil {
		return nil, fmt.Errorf("FromDisk: %w", err)
	}
	return &analyzer{b: b}, nil
}

func (a *analyzer) processText(text string) (*doc.Doc, error) {
	return a.b.Pipe(text)
}

// analyzeTokens prints text / lemma / pos / tag / is_alpha / is_stop
// for the first 10 tokens. Mirrors ComprehensiveSpacyAnalyzer.analyze_tokens.
func (a *analyzer) analyzeTokens(d *doc.Doc) {
	fmt.Println("\n--- 1. Token Level Analysis ---")
	fmt.Printf("%-15s | %-15s | %-10s | %-10s | %-10s | %s\n",
		"TEXT", "LEMMA", "POS", "TAG", "IS_ALPHA", "IS_STOP")
	fmt.Println(strings.Repeat("-", 75))

	ss := d.Vocab.StringStore()
	n := d.NumTokens()
	if n > 10 {
		n = 10
	}
	for i := 0; i < n; i++ {
		tok := d.Tokens[i]
		fmt.Printf("%-15s | %-15s | %-10s | %-10s | %-10t | %t\n",
			truncate(tok.Text, 15),
			truncate(ss.LookupOrEmpty(tok.Lemma), 15),
			ss.LookupOrEmpty(tok.POS),
			ss.LookupOrEmpty(tok.Tag),
			lexflags.IsAlpha(tok.Text),
			tok.IsStop(d.Vocab),
		)
	}
}

// extractEntities walks Tokens[].EntIOB to build entity spans (B-/I-/O
// scheme; 0=missing, 1=I-, 2=O, 3=B-) and prints each with a
// human-readable explanation. Mirrors spaCy's `for ent in doc.ents`.
func (a *analyzer) extractEntities(d *doc.Doc) {
	fmt.Println("\n--- 2. Named Entity Recognition (NER) ---")
	ents := entitySpans(d)
	if len(ents) == 0 {
		fmt.Println("No entities found.")
		return
	}
	ss := d.Vocab.StringStore()
	for _, e := range ents {
		label := ss.LookupOrEmpty(d.Tokens[e.start].EntType)
		fmt.Printf("Entity: %-25s | Label: %-10s | Explanation: %s\n",
			spanText(d, e.start, e.end), label, explain(label))
	}
}

// extractNounChunks uses Doc.NounChunks (gospacy port of
// lang/en/syntax_iterators.noun_chunks). Mirrors `doc.noun_chunks`.
func (a *analyzer) extractNounChunks(d *doc.Doc) {
	fmt.Println("\n--- 3. Noun Chunks ---")
	chunks := d.NounChunks()
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text()
	}
	fmt.Println(strings.Join(texts, " | "))
}

// analyzeDependencies prints text / dep / head.text for the first 10
// tokens. Mirrors ComprehensiveSpacyAnalyzer.analyze_dependencies.
func (a *analyzer) analyzeDependencies(d *doc.Doc) {
	fmt.Println("\n--- 4. Dependency Parsing ---")
	fmt.Printf("%-15s | %-15s | %s\n", "TOKEN", "DEPENDENCY", "HEAD TOKEN")
	fmt.Println(strings.Repeat("-", 50))

	ss := d.Vocab.StringStore()
	n := d.NumTokens()
	if n > 10 {
		n = 10
	}
	for i := 0; i < n; i++ {
		tok := d.Tokens[i]
		head := "<self>"
		if tok.Head >= 0 && tok.Head < d.NumTokens() {
			head = d.Tokens[tok.Head].Text
		}
		fmt.Printf("%-15s | %-15s | %s\n",
			truncate(tok.Text, 15),
			ss.LookupOrEmpty(tok.Dep),
			head,
		)
	}
}

// segmentSentences uses Doc.Sents (reads SentStart populated by the
// parser). Mirrors `for sent in doc.sents`.
func (a *analyzer) segmentSentences(d *doc.Doc) {
	fmt.Println("\n--- 5. Sentence Segmentation ---")
	for i, s := range d.Sents() {
		fmt.Printf("Sentence %d: %s\n", i+1, strings.TrimSpace(s.Text()))
	}
}

// runMatcher uses gospacy's matcher.Matcher (Tier 1, equality-only).
// The two-alternative pattern handles "Artificial Intelligence"
// (multi-token) and bare "AI" (single-token). Same-key overlap dedup
// (longest-first) means the multi-token alt wins when both fire on
// "Artificial Intelligence".
//
// Quantifier OPs (?, *, +) are NOT_YET_PORTED — see Matcher Tier 2.
// For Tier 1 we model "optional Intelligence" as two alternatives.
func (a *analyzer) runMatcher(d *doc.Doc) {
	fmt.Println("\n--- 6. Rule-based Matching ---")

	m := matcher.New(d.Vocab)
	if err := m.Add("AI_PATTERN",
		// Alt 1: "Artificial Intelligence" or "AI Intelligence"
		[]matcher.TokenSpec{
			{LowerIn: []string{"artificial", "ai"}},
			{Lower: "intelligence"},
		},
		// Alt 2: bare "AI" or bare "Artificial"
		[]matcher.TokenSpec{
			{LowerIn: []string{"artificial", "ai"}},
		},
	); err != nil {
		fmt.Printf("Matcher.Add failed: %v\n", err)
		return
	}

	hits := m.Matches(d)
	if len(hits) == 0 {
		fmt.Println("No custom matches found.")
		return
	}
	for _, hit := range hits {
		fmt.Printf("Match Rule: %-12s | Matched Text: %q\n",
			hit.Key, spanText(d, hit.Start, hit.End))
	}
}

// calculateSimilarity computes mean-pooled cosine similarity between
// two texts. Mirrors spaCy's default Doc.similarity (which is also
// mean-of-vectors cosine, ignoring OOV tokens). Needs an md or lg
// bundle — sm has empty vectors and this prints 0.0.
func (a *analyzer) calculateSimilarity(text1, text2 string) {
	fmt.Println("\n--- 7. Document Similarity ---")
	vec := a.b.Vocab.Vectors()
	if vec == nil || vec.Rows() == 0 {
		fmt.Println("Bundle has no vectors loaded (sm-style). Use en_core_web_md or _lg.")
		return
	}

	d1, err := a.b.Pipe(text1)
	if err != nil {
		fmt.Printf("Pipe(text1) failed: %v\n", err)
		return
	}
	d2, err := a.b.Pipe(text2)
	if err != nil {
		fmt.Printf("Pipe(text2) failed: %v\n", err)
		return
	}

	v1 := meanVector(d1, vec)
	v2 := meanVector(d2, vec)
	sim := cosine(v1, v2)

	fmt.Printf("Text 1: %s\n", text1)
	fmt.Printf("Text 2: %s\n", text2)
	fmt.Printf("Similarity Score: %.4f\n", sim)
}

// ---------- helpers ----------

type entSpan struct{ start, end int }

// entitySpans walks d.Tokens[].EntIOB and returns one entSpan per
// contiguous entity. EntIOB scheme: 0 missing, 1 I-, 2 O, 3 B-.
func entitySpans(d *doc.Doc) []entSpan {
	var out []entSpan
	i := 0
	for i < d.NumTokens() {
		if d.Tokens[i].EntIOB != 3 { // not a B- start
			i++
			continue
		}
		start := i
		i++
		for i < d.NumTokens() && d.Tokens[i].EntIOB == 1 { // continue while I-
			i++
		}
		out = append(out, entSpan{start: start, end: i})
	}
	return out
}

// spanText concatenates Token.Text + Whitespace for tokens [start, end),
// trimming the final whitespace. Equivalent to spaCy's Span.text.
func spanText(d *doc.Doc, start, end int) string {
	var b strings.Builder
	for i := start; i < end; i++ {
		b.WriteString(d.Tokens[i].Text)
		if i < end-1 {
			b.WriteString(d.Tokens[i].Whitespace)
		}
	}
	return b.String()
}

// meanVector returns the per-token mean of vectors for the tokens in d
// that have an entry in vec (skips OOV tokens, matching spaCy).
// Returns a zero-length slice if no in-vocab tokens exist.
func meanVector(d *doc.Doc, vec *vocab.Vectors) []float32 {
	cols := vec.Cols()
	sum := make([]float32, cols)
	count := 0
	for i := 0; i < d.NumTokens(); i++ {
		row, ok := vec.Row(d.Tokens[i].Lower)
		if !ok || len(row) != cols {
			continue
		}
		for j := 0; j < cols; j++ {
			sum[j] += row[j]
		}
		count++
	}
	if count == 0 {
		return sum
	}
	inv := 1.0 / float32(count)
	for j := 0; j < cols; j++ {
		sum[j] *= inv
	}
	return sum
}

// cosine returns a · b / (|a| · |b|), or 0 if either is the zero vector.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// truncate cuts s to at most n bytes (no rune-awareness needed; the
// table is decorative). Pads to width via the format string.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

// explain is a small subset of spaCy's spacy.explain() — the OntoNotes
// NER labels emitted by en_core_web_md / lg. Unknown labels return "".
func explain(label string) string {
	switch label {
	case "PERSON":
		return "People, including fictional"
	case "NORP":
		return "Nationalities or religious or political groups"
	case "FAC":
		return "Buildings, airports, highways, bridges, etc."
	case "ORG":
		return "Companies, agencies, institutions, etc."
	case "GPE":
		return "Countries, cities, states"
	case "LOC":
		return "Non-GPE locations, mountain ranges, bodies of water"
	case "PRODUCT":
		return "Objects, vehicles, foods, etc. (Not services.)"
	case "EVENT":
		return "Named hurricanes, battles, wars, sports events, etc."
	case "WORK_OF_ART":
		return "Titles of books, songs, etc."
	case "LAW":
		return "Named documents made into laws"
	case "LANGUAGE":
		return "Any named language"
	case "DATE":
		return "Absolute or relative dates or periods"
	case "TIME":
		return "Times smaller than a day"
	case "PERCENT":
		return `Percentage, including "%"`
	case "MONEY":
		return "Monetary values, including unit"
	case "QUANTITY":
		return "Measurements, as of weight or distance"
	case "ORDINAL":
		return `"first", "second", etc.`
	case "CARDINAL":
		return "Numerals that do not fall under another type"
	}
	return ""
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: comprehensive-analyzer <path/to/en_core_web_md>")
		os.Exit(2)
	}
	modelPath := os.Args[1]

	sampleText := strings.Join([]string{
		"Apple Inc. is considering buying a U.K. startup for $1 billion.",
		"Tim Cook announced the acquisition in London on Tuesday.",
		"Artificial Intelligence will play a massive role in their new software update.",
		"AI is transforming the modern tech landscape incredibly fast.",
	}, " ")

	comparisonText := "A British technology company might be purchased by Apple for one billion dollars."

	a, err := newAnalyzer(modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	d, err := a.processText(sampleText)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Pipe failed: %v\n", err)
		os.Exit(1)
	}

	a.analyzeTokens(d)
	a.extractEntities(d)
	a.extractNounChunks(d)
	a.analyzeDependencies(d)
	a.segmentSentences(d)
	a.runMatcher(d)
	a.calculateSimilarity(sampleText, comparisonText)
}
