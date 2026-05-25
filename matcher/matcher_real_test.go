package matcher_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/bundle"
	"github.com/bioshock/gospacy/v3/matcher"
)

// TestMatcher_RealBundle_StrictDifferential runs every (pattern, text)
// pair from testharness/dump_matcher.py through gospacy and asserts
// the resulting match list is *exactly* identical to Python's. Same
// key, same Start, same End. Strict-100% — any drift is a regression
// to investigate, not a tolerance to widen.
//
// Skipped when en_core_web_sm isn't on disk.
func TestMatcher_RealBundle_StrictDifferential(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..")
	bundlePath := filepath.Join(root, "testdata", "models", "en_core_web_sm")
	if _, err := os.Stat(filepath.Join(bundlePath, "meta.json")); err != nil {
		t.Skipf("en_core_web_sm not present at %s; run testharness/download_assets.sh", bundlePath)
	}
	goldenPath := filepath.Join(root, "testdata", "golden", "matcher_cases.json")
	if _, err := os.Stat(goldenPath); err != nil {
		t.Skipf("golden missing at %s; run testharness/.venv/bin/python testharness/dump_matcher.py", goldenPath)
	}

	var golden struct {
		Patterns []struct {
			Key     string             `json:"key"`
			Pattern [][]map[string]any `json:"pattern"`
		} `json:"patterns"`
		Cases []struct {
			Text    string `json:"text"`
			Matches []struct {
				Key   string `json:"key"`
				Start int    `json:"start"`
				End   int    `json:"end"`
			} `json:"matches"`
		} `json:"cases"`
	}
	data, err := os.ReadFile(goldenPath)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(data, &golden))

	b, err := bundle.FromDisk(bundlePath)
	require.NoError(t, err)

	m := matcher.New(b.Vocab)
	for _, p := range golden.Patterns {
		require.NoErrorf(t, m.FromPatternDict(p.Key, p.Pattern),
			"FromPatternDict(%q)", p.Key)
	}

	for _, c := range golden.Cases {
		t.Run(c.Text, func(t *testing.T) {
			d, err := b.Pipe(c.Text)
			require.NoError(t, err)
			got := m.Matches(d)

			require.Equalf(t, len(c.Matches), len(got),
				"text=%q want %d matches, got %d:\n  want=%+v\n  got =%+v",
				c.Text, len(c.Matches), len(got), c.Matches, got)
			for i, want := range c.Matches {
				require.Equalf(t, want.Key, got[i].Key,
					"text=%q match %d key", c.Text, i)
				require.Equalf(t, want.Start, got[i].Start,
					"text=%q match %d start", c.Text, i)
				require.Equalf(t, want.End, got[i].End,
					"text=%q match %d end", c.Text, i)
			}
		})
	}
}
