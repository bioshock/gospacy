package tokenizer

import (
	"sort"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// Token is one output token. Minimal subset of spacy.Token.
type Token struct {
	Orth string
	Idx  int  // Unicode codepoint (rune) offset in the original text, matching spaCy's token.idx
	WS   bool // true if a whitespace char follows in the original text
	// NormOverride is set only when this token came from a special-case rule
	// that supplied a NORM attribute (e.g. "n't"→"not"). Empty string means
	// "no override". ToDoc reads this to bypass the default Lower fallback.
	NormOverride string
}

// Tokenizer is the language-independent tokenizer engine.
type Tokenizer struct {
	rules *Rules
}

// New returns a Tokenizer driven by rules.
func New(rules *Rules) *Tokenizer { return &Tokenizer{rules: rules} }

// Tokenize splits text into tokens. Walks whitespace-separated chunks
// left-to-right; each chunk recursively split by special-case map,
// prefix, suffix, infix, in that order. A second pass then merges
// consecutive tokens whose concatenated orths match a special case
// (mirroring spaCy's _apply_special_cases retokenization step).
//
// Whitespace handling mirrors spaCy: a single ASCII space (' ') immediately
// after a word token sets WS=true on that token and is consumed silently.
// Any other whitespace run (leading spaces, tabs, newlines, or 2+ spaces
// after a word) is emitted as its own token. When a whitespace run starts
// with a ' ' after a word token, that first space is consumed as WS=true
// and the rest of the run is emitted as a token.
//
// Idx values are Unicode codepoint (rune) offsets, matching spaCy's token.idx.
func (t *Tokenizer) Tokenize(text string) []Token {
	runes := []rune(text)
	var out []Token
	i := 0
	for i < len(runes) {
		if !isSpaceRune(runes[i]) {
			// Non-whitespace: collect the full chunk.
			j := i
			for j < len(runes) && !isSpaceRune(runes[j]) {
				j++
			}
			chunk := string(runes[i:j])
			t.tokenizeChunk(chunk, i, &out)
			i = j
			continue
		}
		// Whitespace run starting at i.
		j := i
		for j < len(runes) && isSpaceRune(runes[j]) {
			j++
		}
		wsRune := runes[i:j]
		n := len(out)
		if n > 0 && wsRune[0] == ' ' {
			// Consume the first space as WS=true on the preceding token.
			out[n-1].WS = true
			remainder := wsRune[1:]
			if len(remainder) > 0 {
				// Emit the leftover whitespace as a token.
				out = append(out, Token{Orth: string(remainder), Idx: i + 1})
			}
		} else {
			// Leading whitespace, or run starting with non-space (tab/newline):
			// emit the entire run as a token.
			out = append(out, Token{Orth: string(wsRune), Idx: i})
		}
		i = j
	}
	// Second pass: merge consecutive tokens whose concatenated orths match a
	// special case. Mirrors spaCy's _apply_special_cases / _special_matcher.
	// Maximum window size is 3 (longest split special-case in en_core_web_sm).
	out = t.applySpecialCases(out)
	return out
}

// ToDoc tokenizes text into a *doc.Doc, interning Orth into v.StringStore.
// Unlike Tokenize (which returns the legacy []Token slice), this is the entry
// point used by bundle.Pipe — it produces the runtime container the pipeline
// components mutate.
//
// Per-token Whitespace is reconstructed from consecutive Idx values plus the
// final tail of text, so Doc.Text() round-trips the input verbatim.
func (t *Tokenizer) ToDoc(v *vocab.Vocab, text string) *doc.Doc {
	flat := t.Tokenize(text)
	d := doc.NewDoc(v, text)
	if len(flat) == 0 {
		return d
	}
	runes := []rune(text)
	d.Tokens = make([]doc.Token, len(flat))
	ss := v.StringStore()
	for i, ft := range flat {
		// Determine the whitespace that follows this token: the runes between
		// the end of ft and the Idx of the next token (or end of text).
		startRune := ft.Idx
		endRune := startRune + len([]rune(ft.Orth))
		nextStart := len(runes)
		if i+1 < len(flat) {
			nextStart = flat[i+1].Idx
		}
		ws := ""
		if endRune < nextStart && endRune <= len(runes) {
			ws = string(runes[endRune:nextStart])
		}
		// If this is the last token, append any trailing whitespace from the
		// source so Doc.Text() round-trips even when input ends in space.
		if i == len(flat)-1 && endRune < len(runes) {
			ws = string(runes[endRune:])
		}
		lex := v.Get(ft.Orth)
		// Norm: defaults to the lexeme's Lower hash. Special-case rules may
		// override this (e.g. clitic "n't" carries NORM:"not" so the NORM
		// hash matches Python's strings["not"], not strings["n't"]).
		norm := lex.Lower
		if ft.NormOverride != "" {
			norm = v.Get(ft.NormOverride).Orth
		}
		d.Tokens[i] = doc.Token{
			Orth:       lex.Orth,
			Lower:      lex.Lower,
			Prefix:     lex.Prefix,
			Suffix:     lex.Suffix,
			Norm:       norm,
			Shape:      ss.LookupOrEmpty(lex.Shape),
			Text:       ft.Orth,
			Whitespace: ws,
			Idx:        ft.Idx,
			SentStart:  -1, // unknown until senter/parser runs
		}
	}
	return d
}

// applySpecialCases merges consecutive tokens when their concatenated orths
// form a key in the specials map, emitting the special-case pieces instead.
// It processes the list left-to-right, trying the largest window first (3),
// so that longer matches take priority over shorter ones.
func (t *Tokenizer) applySpecialCases(toks []Token) []Token {
	if len(toks) < 2 {
		return toks
	}
	result := make([]Token, 0, len(toks))
	i := 0
	for i < len(toks) {
		merged := false
		// Try window sizes from largest to smallest (3, 2).
		for winSize := 3; winSize >= 2; winSize-- {
			if i+winSize > len(toks) {
				continue
			}
			// Check that no whitespace interrupts the window: each token
			// except the last must have WS=false (no space before the next).
			interrupted := false
			for j := i; j < i+winSize-1; j++ {
				if toks[j].WS {
					interrupted = true
					break
				}
			}
			if interrupted {
				continue
			}
			// Concatenate orths.
			key := ""
			for j := i; j < i+winSize; j++ {
				key += toks[j].Orth
			}
			if pieces, ok := t.rules.Special(key); ok {
				// Skip no-op expansions: if the special case output is identical
				// to the input tokens (same orths in the same order), applying it
				// would consume the tokens without changing anything but prevent
				// later passes from finding other merges (e.g. "Wed"→["We","d"]
				// consumes "We"+"d" so "d"+"." can't merge into "d.").
				if isNoOp(pieces, toks[i:i+winSize]) {
					break
				}
				// Emit special case pieces.
				baseIdx := toks[i].Idx
				lastWS := toks[i+winSize-1].WS
				offset := baseIdx
				for pi, p := range pieces {
					ws := false
					if pi == len(pieces)-1 {
						ws = lastWS
					}
					result = append(result, Token{Orth: p.Orth, Idx: offset, WS: ws, NormOverride: p.Norm})
					offset += len([]rune(p.Orth))
				}
				i += winSize
				merged = true
				break
			}
		}
		if !merged {
			result = append(result, toks[i])
			i++
		}
	}
	return result
}

// isNoOp returns true when applying the special case would not change the
// token sequence (the pieces have the same orths as the input tokens, in
// the same order). Skipping no-op merges allows subsequent positions to
// find real merges (e.g. "d" + "." → "d.").
func isNoOp(pieces []SpecialPiece, toks []Token) bool {
	if len(pieces) != len(toks) {
		return false
	}
	for i, p := range pieces {
		if p.Orth != toks[i].Orth {
			return false
		}
	}
	return true
}

// tokenizeChunk implements the spaCy tokenizer algorithm:
//
//  1. Iteratively strip prefixes from the left and suffixes from the right of
//     the chunk, collecting them (mirroring _split_affixes).
//  2. On the middle string, check token_match, special case, url_match, then
//     infix patterns (mirroring _attach_tokens).
//  3. After an infix split, spaCy emits ALL infix-separated substrings in one
//     pass as raw lexemes — not just the leftmost. Parts between infixes are
//     pushed directly without further prefix/suffix/special-case processing.
//  4. Emit: [collected prefixes] + [middle tokens] + [collected suffixes reversed].
func (t *Tokenizer) tokenizeChunk(chunk string, baseIdx int, out *[]Token) {
	if chunk == "" {
		return
	}

	// --- Phase 1: strip affixes iteratively (mirrors _split_affixes) ----------
	type affixToken struct {
		orth string
		idx  int
	}
	var prefixes []affixToken
	var suffixes []affixToken // stored in strip order; emitted in reverse

	middle := chunk
	midIdx := baseIdx
	midRunes := []rune(middle)

	lastSize := -1
	for len(midRunes) > 0 && len(midRunes) != lastSize {
		if t.rules.IsTokenMatch(middle) {
			break
		}
		if _, ok := t.rules.Special(middle); ok {
			break
		}
		lastSize = len(midRunes)

		// Try prefix.
		if pre, ok := t.rules.FindPrefix(middle); ok {
			preRunes := len([]rune(pre))
			prefixes = append(prefixes, affixToken{pre, midIdx})
			midIdx += preRunes
			midRunes = midRunes[preRunes:]
			middle = string(midRunes)
		}

		// Try suffix (on the post-prefix string).
		if suf, ok := t.rules.FindSuffix(middle); ok {
			sufRunes := len([]rune(suf))
			sufIdx := midIdx + len(midRunes) - sufRunes
			suffixes = append(suffixes, affixToken{suf, sufIdx})
			midRunes = midRunes[:len(midRunes)-sufRunes]
			middle = string(midRunes)
		}

		// If we're about to iterate again, re-check whether the middle is
		// now a special case or token match, which would stop stripping.
		if len(midRunes) == lastSize {
			break
		}
	}

	// --- Phase 2: emit prefixes -----------------------------------------------
	for _, p := range prefixes {
		*out = append(*out, Token{Orth: p.orth, Idx: p.idx})
	}

	// --- Phase 3: handle middle (_attach_tokens) ------------------------------
	if middle == "" {
		// nothing to do
	} else if t.rules.IsTokenMatch(middle) {
		*out = append(*out, Token{Orth: middle, Idx: midIdx})
	} else if pieces, ok := t.rules.Special(middle); ok {
		offset := midIdx
		for _, p := range pieces {
			*out = append(*out, Token{Orth: p.Orth, Idx: offset, NormOverride: p.Norm})
			offset += len([]rune(p.Orth))
		}
	} else if t.rules.IsURLMatch(middle) {
		*out = append(*out, Token{Orth: middle, Idx: midIdx})
	} else {
		spans := t.rules.FindInfixes(middle)
		if len(spans) > 0 {
			// Sort spans by start position (ascending), then de-overlap so that
			// spans from different patterns don't produce overlapping splits.
			spans = sortAndDeoverlap(spans)
			// Emit all infix-separated substrings in one pass. spaCy pushes
			// left/infix/.../right directly as raw lexemes without further
			// prefix/suffix/special-case processing on the intermediate parts.
			midR := []rune(middle)
			pos := 0
			for _, sp := range spans {
				if sp.Start > pos {
					*out = append(*out, Token{Orth: string(midR[pos:sp.Start]), Idx: midIdx + pos})
				}
				*out = append(*out, Token{Orth: string(midR[sp.Start:sp.End]), Idx: midIdx + sp.Start})
				pos = sp.End
			}
			if pos < len(midR) {
				*out = append(*out, Token{Orth: string(midR[pos:]), Idx: midIdx + pos})
			}
		} else {
			*out = append(*out, Token{Orth: middle, Idx: midIdx})
		}
	}

	// --- Phase 4: emit suffixes in reverse order ------------------------------
	for i := len(suffixes) - 1; i >= 0; i-- {
		s := suffixes[i]
		*out = append(*out, Token{Orth: s.orth, Idx: s.idx})
	}
}

// sortAndDeoverlap sorts spans by Start and removes any span whose start
// falls inside a previously-seen span (i.e., overlapping spans from
// different infix patterns are resolved by keeping the first one).
func sortAndDeoverlap(spans []Span) []Span {
	sort.Slice(spans, func(i, j int) bool {
		return spans[i].Start < spans[j].Start
	})
	// de-overlap in-place
	n := 0
	end := -1
	for _, sp := range spans {
		if sp.Start >= end {
			spans[n] = sp
			n++
			end = sp.End
		}
	}
	return spans[:n]
}

func isSpaceRune(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\v' || r == '\f'
}
