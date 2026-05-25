# gospacy changelog

## v3.8.14-port.2 (v0.2.1-alpha) — 2026-05-26 (Matcher Tier 1 — equality-only)

Adds a public `matcher` package mirroring `spacy.matcher.Matcher` for
the equality-only subset that covers ~95% of real-world Matcher
usage. No quantifier OPs (Tier 2), no PhraseMatcher / FUZZY (Tier 3) —
both deferred until a consumer files a concrete need. See
`NOT_YET_PORTED.md` for the tier breakdown.

### Added

- **`matcher.Matcher` + `matcher.TokenSpec` + `matcher.Match`** —
  named patterns over `*doc.Doc`. Single-goroutine by gospacy
  convention.

  ```go
  m := matcher.New(b.Vocab)
  m.Add("AI_PATTERN", []matcher.TokenSpec{
      {LowerIn: []string{"artificial", "ai"}},
      {Lower:   "intelligence"},
  })
  for _, hit := range m.Matches(doc) {
      fmt.Println(hit.Key, doc.Tokens[hit.Start:hit.End])
  }
  ```

  Supports ORTH / LOWER / TAG / POS / DEP / LEMMA / ENT_TYPE scalar
  + `{IN}` sets + `{NOT_IN}` (TAG and DEP) + `{REGEX}` on LOWER;
  IS_SPACE / IS_STOP / IS_ALPHA / IS_PUNCT / IS_DIGIT / LIKE_NUM
  tri-state boolean flags. Same-key multi-alternative patterns get
  longest-first overlap dedup; cross-key overlaps preserved.

- **`matcher.Matcher.FromPatternDict(key, [][]map[string]any)`** —
  loads patterns in the Python dict shape spaCy uses on disk and in
  `nlp.add_pipe(...).add_patterns([...])`. Validation is strict:
  unsupported keys, malformed values, and quantifier `OP` all return
  errors (fail-loud per Rule 12).

- **`internal/lexflags`** — pure string→bool helpers exposing
  `IsAlpha`, `IsPunct`, `IsDigit`, `LikeNum` (the latter ports
  `spacy/lang/en/lex_attrs.py:like_num` exactly, including signed
  digits, fractions, cardinal/ordinal words, and `<digits>st|nd|rd|th`
  suffix form). Consumed by the matcher engine and the
  `comprehensive-analyzer` example.

- **`internal/patternspec`** — three pure value-form helpers
  (`ExtractInList`, `ExtractNotInList`, `ExtractRegexString`) factored
  out of the AttributeRuler loader so the new matcher package can
  share them. AR keeps its existing behaviour via thin `var` aliases.

### Changed

- **`examples/comprehensive-analyzer`** — swapped the hand-rolled
  `Token.Lower` scan for the real `matcher.Matcher` (two
  alternatives: `[artificial|ai, intelligence]` and `[artificial|ai]`).
  Output unchanged on the sample text. README's "Three places it
  diverges" table loses the Matcher row; now only Doc.similarity and
  `spacy.explain` remain divergent.

### Tests

- **`matcher/matcher_test.go`** (18 tests) — every attribute axis
  (Orth / Lower / Tag / Pos / Dep / Lemma / EntType / IS_* flags),
  multi-token sequence, alternatives union, overlap dedup
  (same-key longest-first + cross-key preserved), Add validation
  (empty key, conflicting attrs, out-of-range tri-state), Remove.
- **`matcher/loader_test.go`** (5 tests + 1 table-driven of 18
  cases) — round-trip every supported attr × value combo through
  `FromPatternDict`. Quantifier OPs, unknown keys, invalid REGEX,
  non-string IN entries, and wrong-type bool values all fail loud.
- **`matcher/matcher_real_test.go`** — strict-100% Python differential
  against `testharness/dump_matcher.py` on `en_core_web_sm`. 13
  patterns × 8 texts; every match (key + Start + End) identical to
  Python's `spacy.matcher.Matcher`.

- **`internal/lexflags/lexflags_test.go`** — exhaustive coverage of
  `LikeNum` (every Python branch verified) + IsAlpha / IsPunct /
  IsDigit.

### Refactored (internal, no behaviour change)

- AR loader's `extractInList` / `extractNotInList` /
  `extractRegexString` moved to `internal/patternspec`. AR exposes
  them via `var` aliases so existing call sites are unchanged.
  Verified: all AR tests + the real-bundle AR differential still
  green.

## v3.8.14-port.1 (v0.2.0-alpha) — 2026-05-26 (initial public release)

First publishable release. Native Go port of spaCy 3.8.14's English
inference path. Supports `en_core_web_sm`, `en_core_web_md`, and
`en_core_web_lg` bundles loaded directly from the upstream `.spacy`
on-disk format — no Python at runtime.

### What's included

**Annotation pipeline.** `bundle.FromDisk(path)` loads a spaCy bundle and
`Bundle.Pipe(text)` runs the full pipeline: tokenize → tok2vec → tagger
→ parser → attribute_ruler → lemmatizer → NER. Per-token output
(`POS`, `Tag`, `Morph`, `Lemma`, `Head`, `Dep`, `SentStart`, `EntIOB`,
`EntType`) matches Python spaCy exactly on the regression fixtures.

**Selective component skipping.** `Bundle.PipeWith(text, PipeOptions)`
lets parse-only consumers skip lemmatizer / NER / attribute_ruler per
call. Mirrors Python's `nlp.pipe(disable=[...])` /
`with nlp.select_pipes(disable=[...])` semantics.

**Parallel fan-out.** `Bundle.Clone()` deep-copies the mutable
per-bundle state (Vocab, StringStore, lazy pipeline components) while
sharing immutable parameter slices by reference. ~5× cheaper than
re-running `FromDisk`. Documented concurrency contract: a bundle must
be quiescent during `Clone`; each clone is then single-goroutine
thereafter.

**Doc helpers** matching Python's `Token`/`Span` ergonomics:
`Doc.Sents()`, `Doc.NounChunks()`, `doc.ChildrenOf(d, i)`,
`doc.SubtreeOf(d, i)` (CSR-cached for O(1) Children after a one-time
O(N) build), `Token.IsStop(vocab)`.

**Vocab symbol constants.** `vocab.POSNoun`, `vocab.POSPropn`,
`vocab.DepCC`, `vocab.DepConj`, ... — the same integer IDs Python's
`spacy.symbols` exposes. Lets hot loops compare `tok.POS == vocab.POSNoun`
instead of going through a StringStore lookup.

**Neural backend.** `nn/` package with model load/save (msgpack), the
core layer set (Softmax, Maxout, LayerNorm, Residual, Chain, Concatenate,
ExpandWindow, FeatureExtractor, HashEmbed, MultiHashEmbed, PrecomputableAffine,
Tok2VecListener, MaxoutWindowEncoder, StaticVectors, ...), and a gonum-backed
ops implementation (`nn/backend/gonum`). Closure-cached per-layer scratch
buffers cut tok2vec allocation churn ~60%.

**Parser + NER.** Arc-eager transition-based parser
(`spacy.TransitionBasedParser.v2` with `state_type="parser"`) and
BiluoPushDown NER (`state_type="ner"`). Both share the parser's
transition framework. Senter (`state_type="senter"`) is documented as
not-yet-ported — the parser already writes `Token.SentStart`, and
upstream ships senter in `nlp.disabled` by default for sm/md/lg.

**AttributeRuler coverage.** Loader handles every key/value form used
by sm/md/lg's `attribute_ruler/patterns`: `ORTH`/`LOWER`/`TAG`/`DEP`
scalar, `{IN: [...]}`, `{NOT_IN: [...]}` (TAG and DEP), `IS_SPACE`
boolean, and `{REGEX: "..."}` for LOWER. md/lg pattern coverage:
179/179.

**Tokenizer.** English tokenizer port — prefix/suffix/infix rules,
URL/email exception handling, contractions, and the special-case
exceptions from `lang/en/tokenizer_exceptions.py`. Output matches
spaCy character-for-character on the regression fixtures.

**Bundle introspection.** `Bundle.FromDisk` records every pipe declared
in `nlp.pipeline`, including pipes in `nlp.disabled` (recorded as
`Skipped: true` with `SkippedReason: "disabled in nlp.disabled"`, no
build attempt) and pipes whose architecture isn't yet implemented
(recorded as `Skipped` with the build error).

### Out of scope (see NOT_YET_PORTED.md)

Training, beam search, displacy visualisation, `Sentencizer` /
`SentenceRecognizer` (senter) port, `Matcher` / `PhraseMatcher` /
`EntityRuler`, `EntityLinker`, non-English language packs, GPU,
multi-task heads, full BLIS cgo bindings.

### Module path

```
github.com/bioshock/gospacy/v3
```

Go SIV requires the `/v3` suffix since gospacy mirrors upstream's
major version. Future bumps follow `v3.8.14-port.N` — first digit
tracks upstream's tag, `port.N` increments per gospacy release.
