# gospacy changelog

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
