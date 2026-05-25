package nn_test

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/bioshock/gospacy/v3/registry"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
)

func bundleModelPath(t *testing.T, rel string) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "testdata", "models", "en_core_web_sm", rel)
}

// nodeName is the minimal slice of thinc's per-node payload we need to assert
// walk-order. dims/refs/attrs/params are FromBytes's territory.
type nodeName struct {
	Name string `msgpack:"name"`
}

type payloadNames struct {
	Nodes []nodeName `msgpack:"nodes"`
}

func diffStringSlices(got, want []string) string {
	var b strings.Builder
	max := len(got)
	if len(want) > max {
		max = len(want)
	}
	for i := 0; i < max; i++ {
		g, w := "<missing>", "<missing>"
		if i < len(got) {
			g = got[i]
		}
		if i < len(want) {
			w = want[i]
		}
		marker := "  "
		if g != w {
			marker = "!="
		}
		fmt.Fprintf(&b, "%s [%3d] got=%q  want=%q\n", marker, i, g, w)
	}
	return b.String()
}

func TestTok2Vec_BuildMatchesRealBundleWalkOrder(t *testing.T) {
	modelPath := bundleModelPath(t, "tok2vec/model")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("tok2vec model not present at %s; run testharness/download_assets.sh", modelPath)
	}
	raw, err := os.ReadFile(modelPath)
	require.NoError(t, err)
	var p payloadNames
	require.NoError(t, msgpack.Unmarshal(raw, &p))

	cfg := map[string]any{
		"width":                  int64(96),
		"attrs":                  []any{"NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"},
		"rows":                   []any{int64(5000), int64(1000), int64(2500), int64(2500), int64(50), int64(50)},
		"include_static_vectors": false,
		"depth":                  int64(4),
		"window_size":            int64(1),
		"maxout_pieces":          int64(3),
	}
	model, err := registry.Build("spacy.Tok2Vec.v2", cfg)
	require.NoError(t, err)

	walked := model.Walk()
	got := make([]string, len(walked))
	want := make([]string, len(p.Nodes))
	for i, m := range walked {
		got[i] = m.Name
	}
	for i, n := range p.Nodes {
		want[i] = n.Name
	}
	require.Equal(t, len(want), len(got),
		"walk-order length mismatch:\n%s", diffStringSlices(got, want))
	for i := range want {
		require.Equal(t, want[i], got[i],
			"walk-order names differ at index %d:\n%s", i, diffStringSlices(got, want))
	}
}
