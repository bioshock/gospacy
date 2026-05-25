package bundle

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/bioshock/gospacy/v3/doc"
)

// TestBundle_Clone_BasicCorrectness verifies that a cloned Bundle produces
// the same Pipe output as the source on the same input. Sanity check that
// rebuilding the model trees + copying Params preserves correctness.
func TestBundle_Clone_BasicCorrectness(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present at %s", path)
	}
	src, err := FromDisk(path)
	if err != nil {
		t.Fatalf("FromDisk: %v", err)
	}
	clone, err := src.Clone()
	if err != nil {
		t.Fatalf("Clone: %v", err)
	}
	text := "Apple is looking at buying U.K. startup for $1 billion."

	srcDoc, err := src.Pipe(text)
	if err != nil {
		t.Fatalf("src.Pipe: %v", err)
	}
	cloneDoc, err := clone.Pipe(text)
	if err != nil {
		t.Fatalf("clone.Pipe: %v", err)
	}
	if srcDoc.NumTokens() != cloneDoc.NumTokens() {
		t.Fatalf("token count: src=%d clone=%d", srcDoc.NumTokens(), cloneDoc.NumTokens())
	}
	for i := range srcDoc.Tokens {
		s, c := srcDoc.Tokens[i], cloneDoc.Tokens[i]
		if s.Tag != c.Tag || s.POS != c.POS || s.Lemma != c.Lemma ||
			s.Head != c.Head || s.Dep != c.Dep ||
			s.EntIOB != c.EntIOB || s.EntType != c.EntType {
			t.Errorf("tok %d (%q) mismatch:\n  src:   Tag=%d POS=%d Lemma=%d Head=%d Dep=%d EntIOB=%d EntType=%d\n  clone: Tag=%d POS=%d Lemma=%d Head=%d Dep=%d EntIOB=%d EntType=%d",
				i, s.Text,
				s.Tag, s.POS, s.Lemma, s.Head, s.Dep, s.EntIOB, s.EntType,
				c.Tag, c.POS, c.Lemma, c.Head, c.Dep, c.EntIOB, c.EntType)
		}
	}
}

// TestBundle_Clone_ConcurrentPipeNoRace runs N parallel goroutines, each
// with its own Clone, on a variety of texts. Under -race the test FAILS if
// any clone races with another (which would mean the goroutine-per-Bundle
// guarantee was violated). Under default build it sanity-checks throughput
// works at all under load.
//
// Run with: go test -race -run TestBundle_Clone_ConcurrentPipeNoRace ./bundle/
func TestBundle_Clone_ConcurrentPipeNoRace(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present at %s", path)
	}
	src, err := FromDisk(path)
	if err != nil {
		t.Fatalf("FromDisk: %v", err)
	}
	const N = 8
	clones := make([]*Bundle, N)
	for i := range clones {
		c, err := src.Clone()
		if err != nil {
			t.Fatalf("Clone %d: %v", i, err)
		}
		clones[i] = c
	}
	// Diverse texts to force lazy-write code paths (Vocab.Get, StringStore.Add,
	// Lemmatizer.posCache) — exactly the constraints documented in Issue A.
	// Each goroutine processes a unique subset so misses are likely.
	texts := []string{
		"Apple is looking at buying U.K. startup for $1 billion.",
		"The quick brown fox jumps over the lazy dog near the riverbank.",
		"Sentence segmentation, lemmatization, and part-of-speech tagging.",
		"John ran faster than Mary did at the marathon yesterday.",
		"Sales of cryptocurrency exchange services declined last quarter sharply.",
		"Manufacturers of pharmaceuticals must comply with FDA regulations stringently.",
		"Children, animals, and tourists were noticed crossing the busy avenue.",
		"Software for downloadable, electronic, and mobile devices reached new heights.",
		"Microsoft announced new partnerships with Google in machine learning research.",
		"Trademark registration requires careful documentation of all goods and services.",
	}
	var wg sync.WaitGroup
	errs := make(chan error, N*len(texts))
	for i := range clones {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, text := range texts {
				d, err := clones[i].Pipe(text)
				if err != nil {
					errs <- err
					return
				}
				// Sanity touch: read some token data.
				if d.NumTokens() == 0 {
					errs <- &emptyDocErr{i: i, text: text}
					return
				}
				_ = d.Tokens[0].Tag
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("goroutine error: %v", err)
	}
}

type emptyDocErr struct {
	i    int
	text string
}

func (e *emptyDocErr) Error() string {
	return "empty Doc returned from clone " + itoa(e.i) + " on text: " + e.text
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := false
	if i < 0 {
		neg = true
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}

// (Touch doc package so the import is used even when the test compiles
// without the model file being present.)
var _ = doc.Doc{}
