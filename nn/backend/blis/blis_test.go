//go:build blis

package blis

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNew_PhaseScaffold(t *testing.T) {
	o := New()
	require.NotNil(t, o)
}
