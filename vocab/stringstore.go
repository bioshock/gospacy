// Package vocab implements gospacy's lexical infrastructure: the StringStore
// (uint64↔string), Lexeme (per-string lexical attributes), and Vocab (the
// owning container). Most strings use MurmurHash64A with seed=1, matching
// spacy.strings.StringStore (the C++ MurmurHash64A is hard-coded to seed 1
// in spacy/strings.pyx). Linguistic symbols (POS tags, dep labels, morph
// features, etc.) are short-circuited to small fixed IDs via symbolsByStr,
// mirroring spaCy's SYMBOLS_BY_STR table; they do NOT go through murmur.
package vocab

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/bioshock/gospacy/v3/internal/murmur"
)

// StringStore is a bidirectional uint64↔string map. The hash function is
// MurmurHash64A seeded with 1, matching spacy.strings.StringStore. The empty
// string is special-cased to hash 0.
//
// Not safe for concurrent use. The backing keys2str map is unsynchronized;
// Add writes on every previously-unseen non-symbol string. Callers that
// need parallelism must hold one *StringStore (typically one *vocab.Vocab)
// per goroutine.
type StringStore struct {
	keys2str map[uint64]string
}

// NewStringStore returns an empty StringStore. The empty string is pre-installed
// at hash 0.
func NewStringStore() *StringStore {
	return &StringStore{keys2str: map[uint64]string{0: ""}}
}

// Clone returns a deep copy of the StringStore. The new store starts with the
// same hash→string mappings as the source but their backing maps are
// independent: Add on either side does not affect the other. Use this to give
// each goroutine its own StringStore (via vocab.Vocab.Clone, in turn via
// bundle.Bundle.Clone) so concurrent Pipe calls do not race on keys2str.
//
// Safe to call concurrently with reads on the source (Hash / Lookup), but
// NOT safe to call concurrently with Add on the source.
func (s *StringStore) Clone() *StringStore {
	dst := &StringStore{keys2str: make(map[uint64]string, len(s.keys2str))}
	for k, v := range s.keys2str {
		dst.keys2str[k] = v
	}
	return dst
}

// Hash returns the canonical hash for str. For "", it returns 0. For strings
// in the SYMBOLS_BY_STR table (POS tags, dep labels, morph features, etc.) it
// returns the small fixed ID assigned by spaCy. For all other strings it
// returns MurmurHash64A(str, seed=1). Does NOT add str to the store.
func (s *StringStore) Hash(str string) uint64 {
	if str == "" {
		return 0
	}
	if id, ok := symbolsByStr[str]; ok {
		return id
	}
	return murmur.Hash64A([]byte(str), 1)
}

// Add returns the canonical hash for str, interning it in the store when it is
// not a known symbol. For linguistic symbols (see symbolsByStr) the fixed ID is
// returned but the string is NOT stored in keys2str — matching Python's
// behaviour where len(StringStore()) == 0 after s.add("VERB"). Idempotent.
//
// Writes to the unsynchronized keys2str map; not safe for concurrent
// invocation on a shared *StringStore.
func (s *StringStore) Add(str string) uint64 {
	h := s.Hash(str)
	if _, isSymbol := symbolsByStr[str]; isSymbol {
		return h // symbols are implicit; do not clutter keys2str
	}
	if _, ok := s.keys2str[h]; !ok {
		s.keys2str[h] = str
	}
	return h
}

// Get returns the hash for str when it is known to the store. For "", returns
// (0, true). For known symbols (symbolsByStr), returns their fixed ID with
// ok=true without requiring a prior Add. For all other strings ok=false if not
// previously interned via Add. Does NOT auto-intern; use Add for that.
func (s *StringStore) Get(str string) (uint64, bool) {
	if str == "" {
		return 0, true
	}
	if id, ok := symbolsByStr[str]; ok {
		return id, true
	}
	h := s.Hash(str)
	_, ok := s.keys2str[h]
	if !ok {
		return 0, false
	}
	return h, true
}

// Lookup returns the string for a hash. For symbol IDs the name is returned
// without requiring a prior Add call. ok=false if h is neither a symbol ID nor
// a previously interned hash.
func (s *StringStore) Lookup(h uint64) (string, bool) {
	if name, ok := symbolsByID[h]; ok {
		return name, true
	}
	v, ok := s.keys2str[h]
	return v, ok
}

// Len returns the number of non-empty interned strings.
func (s *StringStore) Len() int {
	return len(s.keys2str) - 1 // exclude the always-present ""
}

// Strings returns all interned strings (excluding "") in sorted order.
// Useful for serialization symmetry with Python's StringStore.to_disk.
func (s *StringStore) Strings() []string {
	out := make([]string, 0, len(s.keys2str)-1)
	for _, v := range s.keys2str {
		if v != "" {
			out = append(out, v)
		}
	}
	sort.Strings(out)
	return out
}

// LookupOrEmpty returns Lookup(h) or "" if absent. Convenience for callers
// that want the empty-string fallback (e.g. populating shape from a lexeme).
func (s *StringStore) LookupOrEmpty(h uint64) string {
	v, ok := s.Lookup(h)
	if !ok {
		return ""
	}
	return v
}

// LoadJSON populates the store from a JSON array of strings, the on-disk
// format used by spaCy bundles (vocab/strings.json).
func (s *StringStore) LoadJSON(data []byte) error {
	var arr []string
	if err := json.Unmarshal(data, &arr); err != nil {
		return fmt.Errorf("StringStore.LoadJSON: %w", err)
	}
	for _, str := range arr {
		s.Add(str)
	}
	return nil
}
