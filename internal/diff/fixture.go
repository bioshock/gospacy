// Package diff loads Python-generated golden fixtures from testdata/golden/
// and provides comparators used by Go tests to verify gospacy output against
// the pinned reference spaCy.
package diff

import (
	"encoding/json"
	"fmt"
	"os"
)

// InputFixture is the list of raw input sentences a fixture was generated from.
type InputFixture struct {
	Sentences []string `json:"sentences"`
}

// TokensFixture is the per-sentence tokenizer output dumped by dump_tokens.py.
type TokensFixture struct {
	SpacyVersion string           `json:"spacy_version"`
	Model        string           `json:"model"`
	Sentences    []TokensSentence `json:"sentences"`
}

// TokensSentence holds one sentence's tokenizer output within a TokensFixture.
type TokensSentence struct {
	Text   string  `json:"text"`
	Tokens []Token `json:"tokens"`
}

// Token is one spaCy token in a tokenizer fixture: its text (Orth), byte offset
// (Idx), and whether a trailing space follows it (WS).
type Token struct {
	Orth string `json:"orth"`
	Idx  int    `json:"idx"`
	WS   bool   `json:"ws"`
}

// AttrsFixture is the per-token attribute output dumped by dump_attrs.py.
type AttrsFixture struct {
	SpacyVersion string          `json:"spacy_version"`
	Model        string          `json:"model"`
	Pipeline     []string        `json:"pipeline"`
	Sentences    []AttrsSentence `json:"sentences"`
}

// AttrsSentence holds one sentence's per-token attribute output within an AttrsFixture.
type AttrsSentence struct {
	Text   string      `json:"text"`
	Tokens []TokenAttr `json:"tokens"`
}

// TokenAttr holds the tagger/morphologiser/lemmatiser output for one token.
type TokenAttr struct {
	Orth  string `json:"orth"`
	Tag   string `json:"tag"`
	POS   string `json:"pos"`
	Morph string `json:"morph"`
	Lemma string `json:"lemma"`
}

// ArcsFixture is the per-sentence dep arc list dumped by dump_arcs.py.
type ArcsFixture struct {
	SpacyVersion string         `json:"spacy_version"`
	Model        string         `json:"model"`
	Sentences    []ArcsSentence `json:"sentences"`
}

// ArcsSentence holds one sentence's dependency arc list within an ArcsFixture.
type ArcsSentence struct {
	Text string `json:"text"`
	Arcs []Arc  `json:"arcs"`
}

// Arc is one dependency relation: token index I, its head token index Head, and
// the dependency label Dep (e.g., "nsubj", "dobj").
type Arc struct {
	I    int    `json:"i"`
	Head int    `json:"head"`
	Dep  string `json:"dep"`
}

// EntitiesFixture is the per-sentence NER span list dumped by dump_entities.py.
type EntitiesFixture struct {
	SpacyVersion string             `json:"spacy_version"`
	Model        string             `json:"model"`
	Sentences    []EntitiesSentence `json:"sentences"`
}

// EntitiesSentence holds one sentence's NER span list within an EntitiesFixture.
type EntitiesSentence struct {
	Text     string   `json:"text"`
	Entities []Entity `json:"entities"`
}

// Entity is one named-entity span: token-index range [Start, End) and the NER
// label (e.g., "PERSON", "ORG"). Text is the surface string (informational only).
type Entity struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
	Text  string `json:"text"`
}

func loadJSON(path string, dst any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read fixture %s: %w", path, err)
	}
	if err := json.Unmarshal(b, dst); err != nil {
		return fmt.Errorf("decode fixture %s: %w", path, err)
	}
	return nil
}

// LoadInputFixture reads and decodes an InputFixture from the JSON file at path.
func LoadInputFixture(path string) (*InputFixture, error) {
	var fx InputFixture
	if err := loadJSON(path, &fx); err != nil {
		return nil, err
	}
	return &fx, nil
}

// LoadTokensFixture reads and decodes a TokensFixture from the JSON file at path.
func LoadTokensFixture(path string) (*TokensFixture, error) {
	var fx TokensFixture
	if err := loadJSON(path, &fx); err != nil {
		return nil, err
	}
	return &fx, nil
}

// LoadAttrsFixture reads and decodes an AttrsFixture from the JSON file at path.
func LoadAttrsFixture(path string) (*AttrsFixture, error) {
	var fx AttrsFixture
	if err := loadJSON(path, &fx); err != nil {
		return nil, err
	}
	return &fx, nil
}

// LoadArcsFixture reads and decodes an ArcsFixture from the JSON file at path.
func LoadArcsFixture(path string) (*ArcsFixture, error) {
	var fx ArcsFixture
	if err := loadJSON(path, &fx); err != nil {
		return nil, err
	}
	return &fx, nil
}

// LoadEntitiesFixture reads and decodes an EntitiesFixture from the JSON file at path.
func LoadEntitiesFixture(path string) (*EntitiesFixture, error) {
	var fx EntitiesFixture
	if err := loadJSON(path, &fx); err != nil {
		return nil, err
	}
	return &fx, nil
}
