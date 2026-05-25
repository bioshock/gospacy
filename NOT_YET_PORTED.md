# Not Yet Ported

This file lists every upstream spaCy / thinc feature that gospacy v3.8.14-port.2
**deliberately** does not implement. Each entry names where it would live if/when
it is added, and the rationale for deferring. If you need one of these, open an
issue with your use case — see OVERVIEW.md §6 (anchor users) and §10 (out of
scope).

Phase 7 (v3.8.14-port.2) adds **NER** to the supported pipeline. The supported
shape is now: tokenizer → tagger → parser → attribute_ruler → lemmatizer →
ner. CPU, inference, English, en_core_web_{sm,md,lg}. Everything below is
explicitly outside that line.

---

## Components

### Matcher — Tier 1 shipped, Tier 2 + Tier 3 deferred

- **What:** `spacy.matcher.Matcher` — rule-based token-pattern matcher
  used by `EntityRuler`, custom dependency rules, and many production
  spaCy pipelines.
- **Status:** **Tier 1 (equality-only) shipped in v3.8.14-port.2** as
  the public `github.com/bioshock/gospacy/v3/matcher` package. Covers
  ORTH / LOWER / TAG / POS / DEP / LEMMA / ENT_TYPE scalar + `{IN}` +
  `{NOT_IN}` (TAG / DEP) + `{REGEX}` (LOWER only); IS_SPACE / IS_STOP
  / IS_ALPHA / IS_PUNCT / IS_DIGIT / LIKE_NUM boolean flags; named
  patterns; same-key alternatives; same-key longest-first overlap
  dedup. Multi-token sequence: every TokenSpec matches exactly one
  position.
- **What's deferred:**
  - **Tier 2 — quantifier operators** (`OP: "?" / "*" / "+" / "!" /
    "{n,m}"`). Requires a Thompson-NFA build over token specs;
    ~700 lines + ~15 more Python differential patterns. Defer until
    a consumer files a concrete need.
  - **Tier 3 — `PhraseMatcher`** (trie-backed fast literal phrase
    matching), **`FUZZY`** value form (Levenshtein with token-aware
    edit budget), **`Doc._.foo` extension attrs** (requires a new
    Doc.user_data infrastructure first), numeric comparison ops
    (`==`, `>`, etc. on `LENGTH` / `prob` / `rank`).
- **Where it would live:** `matcher/` (current), with `matcher/nfa.go`
  for Tier 2 and `matcher/phrase.go` / `matcher/fuzzy.go` for Tier 3.
- **Rationale:** 95% of real-world Matcher usage is equality + IN /
  NOT_IN / REGEX. Tier 1 unlocks that for ~1 day of work and reuses
  the AttributeRuler loader machinery (`internal/patternspec`).
  Tier 2/3 land when a concrete consumer hits a quantifier or
  PhraseMatcher gap.

### Entity Linker

- **What:** `spacy.pipeline.entity_linker.EntityLinker` — disambiguate entities
  against a KB.
- **Status:** Not ported. Not in `en_core_web_sm`'s pipeline.
- **Where it would live:** `pipeline/entitylinker.go` (post-NER).
- **Rationale:** Downstream of NER; ports after NER does, if at all.

### Sentence Recognizer (`senter`)

- **What:** `spacy.pipeline.senter.SentenceRecognizer` — neural sentence-boundary
  predictor.
- **Status:** Listed in `en_core_web_sm`'s `config.cfg` pipeline but in
  `nlp.disabled`. gospacy honours `nlp.disabled` and leaves `senter` un-wired.
- **Where it would live:** `pipeline/senter.go` (mirrors `pipeline/tagger.go`'s
  structure — a thin softmax over tok2vec).
- **Rationale:** Off by default in the reference model; the parser already sets
  `Token.SentStart` for the MVP. Add when a bundle ships it enabled.

### Multi-task heads / `incorrect_spans_key`

- **What:** Tagger and parser optional auxiliary heads (`multitasks: [...]`,
  `incorrect_spans_key: <str>`).
- **Status:** Not ported. `parser/cfg` and `tagger/cfg` in `en_core_web_sm`
  carry empty `multitasks: []` and `incorrect_spans_key: null`.
- **Where it would live:** `pipeline/parser.go` `parserCfg` struct + per-doc
  Apply.
- **Rationale:** Training-only feature; this is an inference-only port (OVERVIEW
  §10).

---

## Algorithms

### Beam search (`beam_width > 1`)

- **What:** Non-greedy ArcEager decoding with a beam.
- **Status:** Not ported. `pipeline/parser.go` `NewParser` returns an error
  (`Parser: beam_width=N not supported`) when `beam_width > 1`.
- **Where it would live:** `pipeline/parserinternals/beam.go` (parallel to the
  current greedy `Apply` loop).
- **Rationale:** `en_core_web_sm` ships `beam_width: 1`. The greedy decode
  matches Python 100% on the 8 fixture sentences. Beam adds complexity and a
  10-100x cost for marginal accuracy on the medium-model class.

### Training (oracle, gradients, `learn_tokens`)

- **What:** All learning code — `Parser.set_costs`, `Parser.update`,
  `oracle_actions`, `learn_tokens=True`, gradient computation, backprop.
- **Status:** Not ported. Inference-only.
- **Where it would live:** A future `pipeline/training/` package, if ever.
- **Rationale:** OVERVIEW §10 row 1: "Training. Inference only. Models are
  produced in Python." Out of scope for v1.

---

## Backends

### Full cgo + BLIS bindings

- **What:** Use `cython-blis` for the gemm hot path through a Go cgo wrapper.
- **Status:** Scaffold only. `nn/backend/blis/` exists with build-tag wiring
  (`-tags blis`); the actual cgo bindings are stubs. The pure-Go gonum backend
  is the only one exercised in v0.1.
- **Where it would live:** `nn/backend/blis/{ops.go, ops_cgo.go}` — already
  present as a scaffold.
- **Rationale:** Pure-Go meets the §5 compatibility targets on every shipping
  attribute. cgo+BLIS would close the residual numerical gap and add ~2x on
  gemm, but: it requires libblis-dev at build time, breaks `go install` for
  Windows-no-cgo users, and is a maintenance burden. Ship when there's a real
  perf demand and a real anchor user willing to install libblis.
- **Rationale (updated 2026-05-21, Phase 7 Block B):** Profiling
  `Bundle.Pipe` on a 200-char long-claim-style sentence with
  `en_core_web_sm`, after the B4 lemmatizer-cache fix, shows
  `gonum.blas.sgemmSerial` at 53% cumulative CPU and
  `gonum/internal/asm/f32.DotUnitary` at 32% self time — gemm is now the
  dominant cost. cgo+BLIS would roughly halve those calls based on Phase 2
  per-op ratios, taking the bench from ~3.67 ms/op toward ~2 ms/op. Even
  so, gospacy is already ~2.2× faster than Python on this workload
  (Go 3.67 ms vs Python 8.24 ms end-to-end), so cgo+BLIS is no longer a
  parity requirement — it would be a pure speedup with all of the original
  costs (libblis-dev at build time, cgo barrier, Windows reachability).
  Deferred again. Re-evaluate when an anchor user surfaces a hard latency
  floor that pure-Go cannot meet.

---

## Languages

### Languages other than English

- **What:** German, French, Spanish, Chinese, etc. — every `spacy.lang.<X>`
  module except `lang/en/`.
- **Status:** Not ported. `bundle.FromDisk` returns an error
  (`FromDisk: only nlp.lang=en supported in v1, got %q`) when a non-English
  bundle is loaded.
- **Where it would live:** `lang/<X>/` (mirroring `lang/en/`'s shape:
  `punctuation.go`, `exceptions_gen.go`, `tokenizer_rules.go`) and
  `pipeline/lang/<X>/` (for `IsBaseForm`-style hooks). The `internal/cmd/
  genexceptions` tool generalises with a `--lang` flag.
- **Rationale:** OVERVIEW §10 row 3: "All languages. English first. Others by
  community contribution." Each language is ~3-5 days of work + a differential
  test corpus.

---

## Hardware

### GPU

- **What:** CUDA, Metal, anything non-CPU.
- **Status:** Not ported, no plan. spaCy and thinc themselves are CPU-default
  inference engines; their GPU paths route through CuPy.
- **Where it would live:** Not planned for any phase.
- **Rationale:** OVERVIEW §10 row 2. "CPU only initially." Go's CUDA story is
  weak compared with PyTorch / JAX; adding GPU here would not improve
  throughput at the model sizes we target.

### Non-amd64 architectures (arm64, ppc64le, ...)

- **What:** Tested execution on architectures other than `linux/amd64`.
- **Status:** Source is portable Go; CI is amd64-only.
- **Where it would live:** GitHub Actions matrix expansion.
- **Rationale:** No demand signal yet. PROGRESS.md notes "arm64 deferred (no
  arch-specific Go code yet)" — gonum and msgpack are pure Go and should
  work; "should work" is not "tested".

---

## API surface

### Doc / Token mutating accessors (Python-style attribute setters)

- **What:** spaCy lets you write `token.lemma_ = "be"` to set the lemma. gospacy
  exposes hash fields directly (`tok.Lemma = ss.Add("be")`).
- **Status:** Deliberate Go-idiomatic choice. Documented in OVERVIEW §5
  row 7 ("API ergonomics: idiomatic Go").
- **Rationale:** Not a port goal. Setters add a function-call layer and a
  StringStore allocation pattern that's surprising to Go callers reading hot-
  path code.

### Custom user components

- **What:** `nlp.add_pipe("my_component", ...)` for user-registered Python
  components.
- **Status:** Not supported. `Bundle.Pipe` runs the known sequence (tokenize →
  tagger → parser → attribute_ruler → lemmatizer) only.
- **Where it would live:** A `bundle.Bundle.AddPipe(name string, fn func(*Doc) error)`
  method plus a stable iteration order.
- **Rationale:** Defer until a user asks. The shape of the right API depends
  on the use case (purely additive read-only? mutating? interleaved with
  built-ins?).

---

## Tests & corpora not yet exercised

### Full 10k-sentence parser differential

- **Status:** Phase 5 ran the 8-sentence pipeline_cases differential and got
  100% UAS+DEP. The 10k corpus is downloaded (`testdata/corpora/`) but the
  end-to-end parser-on-corpus differential is not in `make diff-test`.
- **Where it would live:** Extend `testharness/dump_arcs_cases.py` to walk the
  corpus + a Go diff in `pipeline/parser_real_test.go`.
- **Rationale:** Out-of-scope housekeeping for v0.1; the 8-sentence
  differential is the contract. Add when refining the parser perf benchmark.

---

If something here is blocking your use case, please open an issue describing
the workflow and which model bundle you need it for.
