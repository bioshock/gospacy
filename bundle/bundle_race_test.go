//go:build raceverify && race

// Race-reproducer for the Bundle.Pipe single-goroutine constraint
// documented at Bundle.Pipe, vocab.Vocab, vocab.StringStore, and
// pipeline.Lemmatizer.posCache. See OVERVIEW.md §11.
//
// Invocation:
//
//	go test -race -tags=raceverify ./bundle/
//
// EXPECTED OUTCOME: this test FAILS with a DATA RACE warning naming
// Vocab.lexemes, StringStore.keys2str, or Lemmatizer.posCache. The
// failure IS the assertion: it proves the constraint we documented is
// real. If this test ever passes silently, the threading model has
// changed and the godoc on Bundle.Pipe / vocab.Vocab /
// vocab.StringStore must be reviewed and updated to match the new
// guarantee.
//
// The double build tag `raceverify && race` keeps this file out of the
// default build (so `go test ./...` and `go test -race ./...` stay
// green) while guaranteeing that when it does compile, the race
// detector is active and will reliably flag the writes.

package bundle

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

func TestBundle_Pipe_ConcurrentRaceDemonstration(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	path := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(path, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present at %s; run testharness/download_assets.sh", path)
	}
	b, err := FromDisk(path)
	if err != nil {
		t.Fatalf("FromDisk: %v", err)
	}

	// Pre-warm one Pipe call so component initialization (sync.Once) and
	// the first batch of interns happen serially. The race we want the
	// detector to flag is the lazy interning in Vocab.Get /
	// StringStore.Add and the posCache fill in Lemmatizer.Apply on
	// previously-unseen tokens — not the safe sync.Once.
	if _, err := b.Pipe("warmup."); err != nil {
		t.Fatalf("warmup Pipe: %v", err)
	}

	// Each worker emits high-entropy sentences so most token interns are
	// fresh cache misses. Mixing rare lexemes maximizes the race window
	// on Vocab.lexemes / StringStore.keys2str.
	const workers = 8
	const sentencesPerWorker = 32
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for s := 0; s < sentencesPerWorker; s++ {
				text := fmt.Sprintf(
					"Worker %d sentence %d about quokkas antimony phthalocyanine zibellines tarpaulin uvula gibbon zenith.",
					workerID, s,
				)
				if _, err := b.Pipe(text); err != nil {
					t.Errorf("worker %d sentence %d: %v", workerID, s, err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
}
