package vocab

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/vmihailenco/msgpack/v5"
)

// Vectors holds a word-vector matrix and a hash→row index. Mirrors
// spacy.vectors.Vectors in default mode. Empty Vectors (no rows) is valid
// and is the default for en_core_web_sm.
type Vectors struct {
	data    []float32      // flattened (rows*cols) row-major
	cols    int            // dimensionality per row
	key2row map[uint64]int // orth-hash → row index
}

// NumKeys returns len(v.key2row).
func (v *Vectors) NumKeys() int { return len(v.key2row) }

// Rows returns the number of vector rows.
func (v *Vectors) Rows() int {
	if v.cols == 0 {
		return 0
	}
	return len(v.data) / v.cols
}

// Cols returns the per-row dimensionality.
func (v *Vectors) Cols() int { return v.cols }

// Row returns the vector for a given orth hash, or (nil,false) if absent.
// The returned slice is a view into v.data; do not mutate.
func (v *Vectors) Row(hash uint64) ([]float32, bool) {
	if v.cols == 0 {
		return nil, false
	}
	r, ok := v.key2row[hash]
	if !ok {
		return nil, false
	}
	off := r * v.cols
	if off+v.cols > len(v.data) {
		return nil, false
	}
	return v.data[off : off+v.cols], true
}

// RowIndex returns the row index for a given orth hash, or (-1, false) if
// absent. Used by the StaticVectors layer to feed Ints1d into the gemm path.
func (v *Vectors) RowIndex(hash uint64) (int, bool) {
	r, ok := v.key2row[hash]
	if !ok {
		return -1, false
	}
	return r, true
}

// Data returns the flat (rows*cols) float32 backing slice. The caller must not
// mutate the returned slice — it is a view into v.data. Used by the
// StaticVectors layer to perform batched row lookups without re-resolving the
// hash → row mapping inside the layer.
func (v *Vectors) Data() []float32 { return v.data }

// LoadVectorsDir reads vectors + key2row from dir. Accepts the empty-bundle
// shape (vectors file present but matrix has zero rows) used by
// en_core_web_sm. Returns an empty *Vectors with cfg-supplied cols when the
// matrix is absent.
func LoadVectorsDir(dir string) (*Vectors, error) {
	vec := &Vectors{key2row: map[uint64]int{}}

	vecBytes, err := os.ReadFile(filepath.Join(dir, "vectors"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("LoadVectorsDir: read vectors: %w", err)
	}
	if vecBytes != nil {
		if err := vec.decodeVectors(vecBytes); err != nil {
			return nil, fmt.Errorf("LoadVectorsDir: decode vectors: %w", err)
		}
	}

	k2rBytes, err := os.ReadFile(filepath.Join(dir, "key2row"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("LoadVectorsDir: read key2row: %w", err)
	}
	if k2rBytes != nil && len(k2rBytes) > 0 {
		if err := vec.decodeKey2Row(k2rBytes); err != nil {
			return nil, fmt.Errorf("LoadVectorsDir: decode key2row: %w", err)
		}
	}

	return vec, nil
}

// decodeVectors parses the numpy .npy file format used by spaCy to store
// word vectors. The .npy magic is \x93NUMPY followed by version bytes and
// a Python dict header describing shape and dtype.
// An empty shape (0,0) means no vectors; we accept this and return early.
func (v *Vectors) decodeVectors(b []byte) error {
	// Numpy magic: \x93NUMPY
	npyMagic := []byte{0x93, 'N', 'U', 'M', 'P', 'Y'}
	if len(b) < 10 || !bytes.HasPrefix(b, npyMagic) {
		return fmt.Errorf("vectors: not a numpy .npy file (magic mismatch)")
	}
	major := b[6]
	if major != 1 {
		return fmt.Errorf("vectors: unsupported numpy .npy version %d.x", major)
	}
	// header_len is a little-endian uint16 at offset 8.
	headerLen := int(binary.LittleEndian.Uint16(b[8:10]))
	headerEnd := 10 + headerLen
	if headerEnd > len(b) {
		return fmt.Errorf("vectors: header truncated")
	}
	header := strings.TrimSpace(string(b[10:headerEnd]))
	rows, cols, err := parseNpyHeader(header)
	if err != nil {
		return fmt.Errorf("vectors: parse header: %w", err)
	}
	if rows == 0 || cols == 0 {
		return nil // empty bundle — no vectors to load
	}
	v.cols = cols
	dataBytes := b[headerEnd:]
	want := rows * cols * 4 // float32 = 4 bytes
	if len(dataBytes) < want {
		return fmt.Errorf("vectors: data too short: got %d bytes, want %d", len(dataBytes), want)
	}
	v.data = make([]float32, rows*cols)
	for i := range v.data {
		bits := binary.LittleEndian.Uint32(dataBytes[i*4:])
		v.data[i] = math.Float32frombits(bits)
	}
	return nil
}

// parseNpyHeader extracts rows and cols from a numpy header dict string such as:
// {'descr': '<f4', 'fortran_order': False, 'shape': (0, 0), }
func parseNpyHeader(header string) (rows, cols int, err error) {
	// Find 'shape': (...)
	shapeIdx := strings.Index(header, "'shape'")
	if shapeIdx == -1 {
		return 0, 0, fmt.Errorf("no 'shape' key in header: %q", header)
	}
	rest := header[shapeIdx+len("'shape'"):]
	parenOpen := strings.Index(rest, "(")
	parenClose := strings.Index(rest, ")")
	if parenOpen == -1 || parenClose == -1 || parenClose <= parenOpen {
		return 0, 0, fmt.Errorf("malformed shape in header: %q", header)
	}
	shapePart := strings.TrimSpace(rest[parenOpen+1 : parenClose])
	if shapePart == "" {
		return 0, 0, nil // scalar
	}
	dims := strings.Split(shapePart, ",")
	if len(dims) == 1 {
		// 1-D array: (N,) — treat as Nx1
		n, e := strconv.Atoi(strings.TrimSpace(dims[0]))
		if e != nil {
			return 0, 0, fmt.Errorf("bad shape dim: %q", dims[0])
		}
		return n, 1, nil
	}
	r, e1 := strconv.Atoi(strings.TrimSpace(dims[0]))
	c, e2 := strconv.Atoi(strings.TrimSpace(dims[1]))
	if e1 != nil || e2 != nil {
		return 0, 0, fmt.Errorf("bad shape dims: %q", shapePart)
	}
	return r, c, nil
}

func (v *Vectors) decodeKey2Row(b []byte) error {
	var m map[uint64]int64
	if err := msgpack.Unmarshal(b, &m); err != nil {
		return err
	}
	for k, r := range m {
		v.key2row[k] = int(r)
	}
	return nil
}
