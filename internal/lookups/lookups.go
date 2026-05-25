// Package lookups loads msgpack lookup tables in the format written by
// spacy.lookups.Lookups.to_bytes — a map keyed by table name where each value
// is the table's own msgpack-encoded payload. Mirrors spacy.lookups.Lookups
// for the small subset gospacy needs (rule + lookup lemmatization,
// lexeme_norm).
package lookups

import (
	"fmt"
	"os"

	"github.com/vmihailenco/msgpack/v5"
)

// Lookups is a name-keyed bundle of Tables. Mirrors spacy.lookups.Lookups.
type Lookups struct {
	tables map[string]*Table
}

// Table is a uint64→any map. Keys are pre-hashed via spaCy's StringStore
// (MurmurHash64A seed=1); values are msgpack-decoded as-is (list, dict,
// string, int, ...). Callers cast values according to their schema —
// "lemma_rules" yields []any-of-[]any-of-string, "lemma_exc" yields a
// map[string]any-of-[]any-of-string, "lexeme_norm" yields a map of hash→hash.
type Table struct {
	Name string
	Data map[uint64]any
}

// Load reads lookups from path. Accepts both the lemmatizer/lookups/lookups.bin
// layout (multiple tables) and the vocab/lookups.bin layout (single
// lexeme_norm table); both are top-level dicts keyed by table name.
//
// Adaptation from plan: the actual on-disk format produced by spaCy's
// Lookups.to_bytes() serializes self._tables directly, where Table objects
// serialize as their underlying OrderedDict (uint64→any). The result is a flat
// map[string]map[int]any — not a map[string]bytes of nested Table.to_bytes().
// The plan's nested-msgpack-bytes path is provided as a fallback for any future
// bundles that do embed per-table bytes.
func Load(path string) (*Lookups, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("lookups.Load: %w", err)
	}

	// Primary path: outer is map[string]RawMessage. Each inner RawMessage
	// may be either:
	//   (a) a uint64-keyed dict (the actual en_core_web_sm format), or
	//   (b) a {name, dict, bloom} struct (hypothetical nested Table.to_bytes).
	var outer map[string]msgpack.RawMessage
	if err := msgpack.Unmarshal(b, &outer); err != nil {
		return nil, fmt.Errorf("lookups.Load: decode outer: %w", err)
	}

	tables := make(map[string]*Table, len(outer))
	for name, raw := range outer {
		// Try (a) first: raw is the table dict directly (int keys → any).
		var direct map[any]any
		if err := msgpack.Unmarshal(raw, &direct); err == nil && looksLikeHashMap(direct) {
			dict, err := coerceToHashMap(direct)
			if err != nil {
				return nil, fmt.Errorf("lookups.Load: table %q coerce: %w", name, err)
			}
			tables[name] = &Table{Name: name, Data: dict}
			continue
		}

		// Try (b): raw is {name: str, dict: {hash: val}, bloom: bytes}.
		var inner map[string]any
		if err := msgpack.Unmarshal(raw, &inner); err != nil {
			return nil, fmt.Errorf("lookups.Load: decode table %q: %w", name, err)
		}
		dictAny, ok := inner["dict"]
		if !ok {
			return nil, fmt.Errorf("lookups.Load: table %q has no 'dict' key", name)
		}
		dict, err := coerceToHashMap(dictAny)
		if err != nil {
			return nil, fmt.Errorf("lookups.Load: table %q dict: %w", name, err)
		}
		tables[name] = &Table{Name: name, Data: dict}
	}
	return &Lookups{tables: tables}, nil
}

// looksLikeHashMap returns true if m is non-empty and all keys are numeric
// (the table-dict case), or if it's empty (ambiguous but safe to treat as
// hash-map).
func looksLikeHashMap(m map[any]any) bool {
	for k := range m {
		switch k.(type) {
		case int8, int16, int32, int64,
			uint8, uint16, uint32, uint64, int:
			return true
		default:
			return false
		}
	}
	return true // empty map
}

// Has reports whether a table is present.
func (l *Lookups) Has(name string) bool { _, ok := l.tables[name]; return ok }

// Get returns the named table or nil.
func (l *Lookups) Get(name string) *Table { return l.tables[name] }

// Names returns every table name (unsorted).
func (l *Lookups) Names() []string {
	out := make([]string, 0, len(l.tables))
	for k := range l.tables {
		out = append(out, k)
	}
	return out
}

// GetByHash returns the value at hash and whether it was present.
func (t *Table) GetByHash(h uint64) (any, bool) {
	if t == nil {
		return nil, false
	}
	v, ok := t.Data[h]
	return v, ok
}

// Len returns the number of entries.
func (t *Table) Len() int {
	if t == nil {
		return 0
	}
	return len(t.Data)
}

func coerceToHashMap(v any) (map[uint64]any, error) {
	switch x := v.(type) {
	case map[uint64]any:
		return x, nil
	case map[any]any:
		out := make(map[uint64]any, len(x))
		for k, vv := range x {
			h, err := toUint64(k)
			if err != nil {
				return nil, err
			}
			out[h] = vv
		}
		return out, nil
	case map[int64]any:
		out := make(map[uint64]any, len(x))
		for k, vv := range x {
			out[uint64(k)] = vv
		}
		return out, nil
	}
	return nil, fmt.Errorf("dict not a uint64-keyed map: %T", v)
}

func toUint64(v any) (uint64, error) {
	switch x := v.(type) {
	case uint64:
		return x, nil
	case uint32:
		return uint64(x), nil
	case uint16:
		return uint64(x), nil
	case uint8:
		return uint64(x), nil
	case int64:
		return uint64(x), nil
	case int32:
		return uint64(x), nil
	case int16:
		return uint64(x), nil
	case int8:
		return uint64(x), nil
	case int:
		return uint64(x), nil
	}
	return 0, fmt.Errorf("not a uint64 key: %T(%v)", v, v)
}
