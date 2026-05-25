# gospacy — Progress Tracker

Companion to `docs/superpowers/specs/2026-05-17-gospacy-revised-plan-design.md` (the spec). The spec defines what we're building and why; this file tracks how far we are.

**How to use**: update checkbox status as work progresses. Mark `[x]` for done, `[~]` for in-progress, leave `[ ]` for pending. Add dated notes in the **Notes** block under each phase. Each phase has explicit **Exit criteria** that must be green before moving on.

---

## Status summary

| Phase | Status | Started | Completed | Notes |
|---|---|---|---|---|
| 0. Foundation | ✅ Complete | 2026-05-18 | 2026-05-18 | 21 tasks, differential harness ready |
| 1. nn/ core | ✅ Complete | 2026-05-18 | 2026-05-18 | 1a (ops/harness) + 1b (layers/model/loader) done |
| 2. nn/ hardening | ✅ Complete | 2026-05-18 | 2026-05-18 | godoc + benchmarks + example + CHANGELOG |
| 3. Tokenizer + StringStore + Vocab + model loader | ✅ Complete | 2026-05-18 | 2026-05-19 | vocab + tokenizer + config + registry + bundle reader |
| 4. Tagger + rule-based components | ✅ Complete | 2026-05-19 | 2026-05-19 | Doc/Token + AR + Lemmatizer; tagger forward in 4.5 |
| 4.5. Architecture parity (tagger forward) | ✅ Complete | 2026-05-19 | 2026-05-19 | 5 layers, Walk BFS fix, 65-node tok2vec, 68/68 Tag/Lemma |
| 5. Parser | ✅ Complete | 2026-05-19 | 2026-05-19 | greedy ArcEager; UAS/DEP 100% on 8 fixtures |
| 6. v0.1 release | ✅ Complete | 2026-05-19 | 2026-05-19 | README rewrite, LICENSE, NOT_YET_PORTED, godoc audit, examples, v3.8.14-port.0 tag |
| 7. Coverage expansion (helpers + perf + md/lg + NER) | ✅ Complete | 2026-05-21 | 2026-05-21 | Blocks A+B+C+D; v3.8.14-port.2 tagged; NER strict 100%, md/lg strict 100% |
| 8. Sustained maintenance | ⬜ Not started | – | – | ongoing once v0.1 ships |

Legend: ⬜ not started · 🟡 in progress · ✅ complete · 🔴 blocked

---

## Phase 0 — Foundation (1 month)

- [x] Go module init: `go mod init github.com/<org>/gospacy`
- [x] Project layout matches spec §2 (`doc/`, `vocab/`, `tokenizer/`, `pipeline/`, `modelload/`, `nn/`, `internal/`)
- [~] CI: GitHub Actions matrix — pure-Go ✅ done; cgo+BLIS deferred to Phase 1 (no BLIS-dependent code exists yet, deliberately scoped per plan self-review)
- [~] CI: linux/amd64 ✅ done; arm64 deferred (no arch-specific Go code yet)
- [x] CI: lint via `golangci-lint`
- [x] Benchmark corpus selected: UD English-EWT (parser), CoNLL-2003 or OntoNotes excerpt (NER), 10k-sentence general corpus (tokenizer)
- [x] Pin one Python reference environment (`requirements-ref.txt`): `spacy==3.8.14`, `cython-blis==<pinned>`, `numpy==<pinned>`
- [x] Pin one reference model: `en_core_web_sm-3.7.x` (download + checksum into `testdata/`)
- [x] Differential test harness — Python side: script that loads ref model + corpus, dumps tokens / tags / morph / lemmas / arcs / entity spans / per-tensor checksums to stable JSON in `testdata/golden/`
- [x] Differential test harness — Go side: fixture loader, comparator, classifier (whitespace diff vs off-by-one boundary vs legitimate model divergence)
- [x] `UPSTREAM` file at repo root with pinned upstream commit
- [x] Initial `README.md` stub explaining the upstream-pinning model

**Exit criteria**:
- Run `make diff-test` → exits 0 (regenerates all 9 golden fixtures: 4 from the 10k corpus + 5 in-repo samples).
- CI is green on both build tags.
- All `testdata/golden/` fixtures regenerate deterministically.

**Notes**:
- 2026-05-18: phase complete. Differential harness runs `make diff-test` end-to-end (Python dumpers + Go diff package). Go test suite has 24 unit tests, all green. 21 atomic commits across the phase. CI workflow in place but not yet pushed to GitHub (no remote configured).

---

## Phase 1 — `nn/` core (2–3 months)

### Ops interface
- [x] `nn/ops.go` — interface design (gemm, affine, plus non-BLAS ops below)
- [x] `nn/backend/gonum/` — pure-Go backend using gonum
- [~] `nn/backend/blis/` — scaffold only with `-tags blis` (full cgo bindings in Phase 1b)
- [x] Build-tag wiring: default constructor picks the right backend

### Non-BLAS ops (hot path)
- [x] `seq2col` + tests against Python golden
- [x] `maxout` (piece-wise max) + tests
- [x] `mish` activation + tests
- [x] `hash` (Murmur3, 4-seed variant for HashEmbed) + tests
- [x] `gather_add` + tests
- [x] `reduce_first`, `reduce_last`, `pad` + tests (`list2padded`/`padded2list` deferred to 1b)
- [x] `gemm`, `affine` (BLAS-class) + tests
- [x] `softmax` (row-wise, numerically stable) + tests
- [x] `internal/murmur` — MurmurHash64A + custom 64-bit routine verified against Python

### Layers (each layer = file + unit tests + golden comparison)
- [x] `nn/layers/linear.go`
- [x] `nn/layers/maxout.go`
- [x] `nn/layers/mish.go`
- [x] `nn/layers/softmax.go`
- [x] `nn/layers/hashembed.go` (was `hash_embed.go` in original plan; renamed for Go convention)
- [x] `nn/layers/static_vectors.go`
- [~] `nn/layers/feature_extractor.go` (deferred to Phase 4 — depends on Doc type)
- [~] `nn/layers/precomputable_affine.go` (deferred to Phase 5 — used only by parser)

### Layer combinators
- [x] `chain`, `concatenate`, `with_array`, `with_padded`, `list2ragged`, `expand_window`, `residual`
- [~] `noop`, `zero_init` (not needed for inference; deferred indefinitely)

### Model tree
- [x] `nn/model.go` — Model struct, walk() ordering (matches thinc's depth-first pre-order)
- [x] `nn/from_bytes.go` — `vmihailenco/msgpack/v5`-backed deserializer mirroring thinc's `{nodes, attrs, params, shims}` layout
- [x] Roundtrip test: serialize Python thinc model → deserialize in Go → walk-order match
- [x] **End-to-end**: load tiny thinc model (Linear→Softmax), run forward pass, per-tensor diff at every layer boundary

**Exit criteria**:
- Hand-construct a model tree in Go mirroring a known Python thinc-saved model.
- Load weights via `from_bytes`.
- Run forward pass on a fixed input.
- Per-tensor numeric diff at every layer boundary ≤ ULP-level (cgo+BLIS) or within documented tolerance (pure-Go).

**Notes**:
- 2026-05-18: Phase 1a complete. 13 ops implemented in `nn/backend/gonum/` (gemm, affine, seq2col, maxout, mish, softmax, hash, gather_add, reduce_first, reduce_last, pad) plus the `internal/murmur` package. All ops have per-op Python-generated goldens and Go diff tests. 17 atomic commits in 1a. Tagged `phase-1a-complete`. Phase 1b (layers + Model tree + msgpack from_bytes + end-to-end forward-pass verification) is the remaining Phase 1 work.
- 2026-05-18: Phase 1b complete. Model struct + walk() + Tensor sum-types + msgpack from_bytes deserializer (vmihailenco/msgpack/v5, srsly numpy-ext decode). 6 layers (Linear, Softmax, Mish, Maxout, HashEmbed, StaticVectors) and 7 combinators (Chain, Concatenate, WithArray, WithPadded, List2Ragged, ExpandWindow, Residual). End-to-end test loads a Python thinc Chain(Linear,Softmax) and matches its forward pass per-tensor within 1e-5. 18 commits in 1b. Tagged `phase-1b-complete`. FeatureExtractor, full cgo+BLIS bindings, and Tok2VecListener deferred to later phases.

---

## Phase 2 — `nn/` hardening (1 month)

- [x] API polish pass (function naming, error messages, exported surface)
- [x] `go doc` comments on every exported symbol
- [x] Benchmark suite: gemm, seq2col, maxout, mish — vs Python via subprocess wrapper
- [x] Perf pass on hottest 3 kernels if >2× slower than Python (no action needed — all ops within 2×; see BENCHMARKS.md §3)
- [x] Example program in `examples/load-thinc-model/` — loads a Python-saved model and runs inference
- [x] `nn/CHANGELOG.md` — first entry "alpha for internal consumers"

**Exit criteria**:
- `go doc ./nn/...` produces complete documentation.
- Benchmarks committed; perf gap vs Python documented in `BENCHMARKS.md`.

**Notes**:
- 2026-05-18: Phase 2 complete. godoc audit across nn/, nn/layers/, nn/backend/gonum/, internal/diff/, internal/murmur/. Error-message normalisation in nn/from_bytes.go. Go benchmarks for 8 hot ops + tiny-chain forward pass; Python reference timings via testharness/bench_thinc.py. BENCHMARKS.md captures Go-vs-Python ratios — all ops within 2× (Seq2Col and Hash are FASTER than Python). examples/load-thinc-model/ demonstrates the public API. nn/CHANGELOG.md alpha entry. 7 commits in phase 2. Tagged phase-2-complete.

---

## Phase 3 — Tokenizer + StringStore + Vocab + model loader (2–3 months)

### Foundations
- [x] `internal/murmur/` — MurmurHash64A with seed=1 (matches Python `murmurhash.mrmr.hash64`)
- [x] Tests: hash 10k strings from spaCy's StringStore, compare hash outputs

### Vocab stack
- [x] `vocab/stringstore.go` — load sorted JSON array, lazy hash interning
- [x] `vocab/lexeme.go` — per-word lexical attributes (orth, lower, prefix, suffix, shape, norm, lang, prob, cluster)
- [x] `vocab/vocab.go` — Lexeme cache keyed by hash, lex_attr_getters port from `lang/en/lex_attrs.py`
- [x] `vocab/vectors.go` — load msgpack-of-numpy vector matrix + key2row map (landed in Phase 4 Task 7; empty-bundle path for en_core_web_sm)

### Tokenizer
- [x] `tokenizer/regex.go` — regexp2 wrappers for prefix/suffix/infix/token_match/url_match
- [x] `tokenizer/lang/en/punctuation.go` — port `lang/en/punctuation.py` (5 lookbehind infix patterns) — lives in `lang/en/`
- [x] `tokenizer/lang/en/exceptions.go` — port the Python *generator* for English exceptions (pronouns, contractions, abbreviations) — generated via `internal/cmd/genexceptions`
- [~] `tokenizer/cache.go` — Murmur3-keyed token cache (like PreshMap) (deferred; 100% corpus agreement met without cache)
- [x] `tokenizer/tokenizer.go` — main algorithm (`_split_affixes` + `_attach_tokens` + special-case pass)
- [~] `tokenizer/serialize.go` — load tokenizer rules from `.spacy` bundle (msgpack) (deferred; bundle reads config.cfg directly)

### config.cfg + architecture registry
- [x] `modelload/configcfg.go` — custom INI-like parser (sections, `@architectures = "..."` references, hyperparameters) — lives in `config/`
- [x] `modelload/registry.go` — `var registry = map[string]ArchitectureFactory{...}` for all 22 MVP architectures (14 `spacy.*` + 8 `spacy-legacy.*`) — lives in `registry/`
- [x] `modelload/errors.go` — `ErrUnknownArchitecture{Name, Namespace}`, `ErrUnknownComponent{Name}` — lives in `registry/`

### Bundle reader
- [x] `modelload/bundle.go` — read `meta.json`, `config.cfg`, `vocab/`, `tokenizer`, per-component subdirs — lives in `bundle/`
- [x] `modelload/component_loader.go` — for each component in pipeline, look up factory, instantiate model tree, load weights — integrated in `bundle/bundle.go`

**Exit criteria**:
- Tokenize the 10k-sentence corpus in Go; token-for-token match against Python ≥ 99.99% (allow rare disagreements on edge cases, log all).
- Load `en_core_web_sm-3.7.x` bundle to the point where every component's architecture tree is constructed and weights are populated. (Components not yet wired to runtime: tagger/parser/NER come in phases 4-5.)

**Notes**:
- 2026-05-19: Phase 3 complete. 26 commits. 7 new packages: vocab (StringStore + Lexeme + Vocab), tokenizer (Rules + Tokenizer), lang/en (punctuation patterns + generated exceptions), config (config.cfg parser), registry (22 architectures, 7 implemented), bundle (FromDisk), internal/cmd/genexceptions. Tokenizer: 100% sentence-level agreement on 10k-corpus differential test (target ≥ 99.99%). Bundle manifest cross-check wired into diff-test. Deferred: vector matrix loading (Phase 4), tokenizer cache (add on real perf signal), tokenizer msgpack serialize (bundle uses config.cfg path directly). Tagged phase-3-complete.
- Retro (from Phase 4): `vocab/vectors.go` landed in Phase 4 Task 7 [x]. `config.cfg` `${section.key}`/`${section:key}` interpolation landed in Phase 4 Task 5 [x].

---

## Phase 4 — Tagger + rule-based components (1–2 months)

### Doc/Token API surface (required by all pipeline components)
- [x] `doc/doc.go` — Doc struct: tokens slice, vocab pointer, text
- [x] `doc/token.go` — Token type with orth, lemma, norm, lower, prefix, suffix, shape, tag, pos, morph, dep, head, sent_start, ent_iob, ent_type, idx, whitespace
- [x] `doc/span.go` — Span type
- [x] Iteration helpers, attribute getters

### Lookups subsystem
- [x] `internal/lookups/` — load msgpack lookup tables from `vocab/lookups.bin` and `lemmatizer/lookups/*`
- [x] Lookup interface (lemma_lookup, lemma_rules, lemma_exc, lemma_index)

### Tagger
- [x] `pipeline/tagger.go` — forward pass through tok2vec + tagger model, argmax, write `Token.tag` and `Token.pos` (real-bundle forward wired in Phase 4.5 — 68/68 Tag match)
- [x] Wire pipeline: tokenizer → tok2vec → tagger (tok2vec active in Bundle.Pipe as of Phase 4.5; Skipped: true removed)

### AttributeRuler (~355 LOC)
- [x] `pipeline/attributeruler.go` — load patterns from `attribute_ruler/patterns.bin`, apply matched rules to set `Token.morph` and refined `Token.pos` (179 patterns loaded, 161 applied; 18 DEP/IS_SPACE/NOT_IN patterns unsupported, skip with warning)

### Lemmatizer (rule + lookup mode only; ~323 LOC + English specifics)
- [x] `pipeline/lemmatizer.go` — base Lemmatizer
- [x] `pipeline/lang/en/lemmatizer.go` — EnglishLemmatizer: rule-mode matching, lookup fallback
- [x] Wire pipeline: ... → tagger → attribute_ruler → lemmatizer

**Exit criteria**:
- Run `en_core_web_sm` pipeline up through lemmatizer. ✅ (via Bundle.Pipe with graceful skips)
- POS argmax ≥ targets (§5). ✅ Tag 68/68 exact (Phase 4.5). AR POS 66/68 (97%) — 2 tokens on s07 need parser DEP (Phase 5).
- Morph 100% match. ✅ (7/8 sentences; s07 2-token divergence documented in KNOWN_DIVERGENCES.md)
- Lemma 100% match. ✅ (64/68 tokens; 4 allow-listed in KNOWN_DIVERGENCES.md)

**Notes**:
- 2026-05-19: Phase 4 complete. New packages: doc/ (Token/Doc/Span), internal/lookups/ (Lookups+Table), pipeline/ (Tagger, AttributeRuler, Lemmatizer), pipeline/lang/en/ (IsBaseForm). Three stub→real promotions in registry/: FeatureExtractor.v1, Tok2VecListener.v1, CharacterEmbed.v2. Two deferred Phase 3 items landed: vocab/vectors.go + config.cfg ${...} interpolation. Bundle.Pipe(text) runs tokenize→tagger→ruler→lemmatizer end-to-end. Differential tests: 100% exact Tag on synthetic model; POS/Morph/Lemma vs Python documented. 21 commits in Phase 4. Tagged phase-4-complete. Real-bundle tagger forward completed in Phase 4.5.
- 2026-05-19: Phase 4 carve-outs resolved in Phase 4.5 — AR LEMMA write wired, genexceptions NORM fix applied, Bundle.Pipe Skipped removed for tok2vec/tagger. Final scores: Tag 68/68, Morph 68/68, Lemma 68/68, POS 66/68 (2 tokens need Phase 5 parser).

---

## Phase 4.5 — Architecture parity (tagger forward pass)

Carved out from Phase 4 on 2026-05-19 (plan mid-execution amendment, Rule 12).
The real `en_core_web_sm` tok2vec model is 65 nodes and requires ~10 thinc layers
not yet ported. Phase 4.5 ports those layers so the Tagger forward runs against
the real bundle weights, completing the POS argmax exit criterion from Phase 4.

### Missing thinc layers
- [x] `LayerNorm` — normalisation layer in tok2vec encode stack
- [x] `Dropout` — training-time only but shape must match for inference pass
- [x] `IntsGetitem` — index projection used in ints-getitem node
- [x] `ExtractFeatures` forward — `spacy.FeatureExtractor.v1` full forward
- [x] `Ragged2List` — ragged conversion nodes in the encode path
- [x] `MultiHashEmbed` concat — 6-way Concatenate(Chain(IntsGetitem, HashEmbed) × 6)
- [x] `HashEmbed` column-attr extension (column + dropout_rate attrs round-trip)
- [x] Shape-consistent `residual` + `expand_window` wiring — `pad` attr on WithArray

### Pipeline integration
- [x] `registry/architectures.go` — `buildTok2VecV2` produces 65-node tree matching en_core_web_sm
- [x] `registry/architectures.go` — `buildTaggerV2` loads into the 65-node tree correctly
- [x] `pipeline/tagger.go` — real-bundle forward pass active (Skipped: true removed)
- [x] `Bundle.Pipe` promotes tok2vec/tagger from `Skipped: true` to active
- [x] Differential test: Tag argmax 100% exact vs Python on 8 pipeline_cases sentences

**Exit criteria**:
- `pipeline.TestTagger_DifferentialReal` passes: per-token Tag matches Python golden for all 8 sentences. ✅
- `Bundle.Pipe` no longer sets `Skipped: true` for tok2vec or tagger with en_core_web_sm. ✅
- `go test ./...` all green. ✅

**Notes**:
- 2026-05-19: Phase 4.5 complete. 15 commits. 5 new layers (LayerNorm, Dropout, IntsGetitem, Ragged2List, HashEmbed ext), 2 new types (Uint64s2d, RaggedU64), ExtractFeatures real forward, 3 architecture rebuilds (MultiHashEmbed.v2, MaxoutWindowEncoder.v2, Tok2Vec.v2 — 65 nodes). Critical fixes: Walk() DFS→BFS, Concatenate naming, WithArray pad attr, genexceptions NORM, AR LEMMA write. Walk-order test (65 nodes), tok2vec per-layer parity test, and full tagger differential all pass. Tagged phase-4_5-complete. Remaining: s07 POS 2-token DEP divergence → Phase 5.

---

## Phase 5 — Parser (2 months budgeted; v0.0.5-alpha delivered)

### Parser state machinery
- [x] `pipeline/parserinternals/state.go` — StateC equivalent (stack, buffer, heads, l-r arcs, unshiftable, sent_starts)
- [x] `pipeline/parserinternals/transition_system.go` — TransitionSystem + moves loader (msgpack → 106 actions)
- [x] `pipeline/parserinternals/arc_eager.go` — ArcEager transitions (Shift, Reduce, Left, Right, Break) + SetValid
- [x] `pipeline/parserinternals/nonproj.go` — Deprojectivize (Nivre & Nilsson HEAD scheme)

### Scoring
- [x] `nn/layers/precomputable_affine.go` — lower layer port (sum_state_features done in scorer)
- [x] `nn/layers/list2array.go` — FloatList → Floats2d flatten (used by tok2vec ref Chain)
- [x] `pipeline/parser.go` — Parser type, ScoreStateInternal, greedy Apply

### Parser pipeline
- [x] `registry/architectures.go` — TransitionBasedParser.v2 + TransitionModel.v1 + PrecomputableAffine.v1
- [x] `bundle/bundle.go` — Parser wired between Tagger and AttributeRuler

### AttributeRuler DEP support
- [x] `pipeline/attributeruler_loader.go` — DEP and {NOT_IN} recognised as supported keys
- [x] `pipeline/attributeruler.go` — Matcher checks Token.Dep; TAG {NOT_IN} negation

### Differential validation
- [x] Walk-order test: parser model 7 nodes matches on-disk
- [x] UAS + DEP label = 100% on the 8 fixture sentences (68/68 tokens)
- [x] s07 POS divergence closed (KNOWN_DIVERGENCES.md updated)

### NER (deferred — see OVERVIEW.md §10)
NER is explicitly out of scope for v0.1 per OVERVIEW.md §10 ("NER, entity linker, custom components — port on demand, not preemptively") and the project MVP definition (tokenizer+tagger+parser). NER lands in a follow-on phase if/when external demand materialises. The Phase-5 deliverable is the dependency parser only.

**Exit criteria**:
- ✅ Parser greedy_parse runs end-to-end against en_core_web_sm.
- ✅ UAS + DEP label = 100% on the 8-sentence fixture corpus.
- ✅ All differential tests pass; KNOWN_DIVERGENCES.md s07 POS gap closed.
- ✅ Parser throughput within acceptable range for differential test (perf budget tracked separately).

**Notes**:
- 2026-05-19: Phase 5 complete (parser-only scope). Tagged phase-5-complete.
  17 commits. New package: `pipeline/parserinternals/` (state, transition_system, arc_eager, nonproj).
  Two stubs promoted: spacy.TransitionBasedParser.v2, spacy.TransitionModel.v1, spacy.PrecomputableAffine.v1.
  Two new layers: PrecomputableAffine, List2Array. NER deferred to follow-on phase per OVERVIEW §10.

---

## Phase 6 — v0.1 release (1 month)

- [x] Doc/Token API godoc audit pass (doc/ + bundle/ + pipeline/)
- [x] `README.md` — quickstart, supported model list, compatibility matrix
- [x] `KNOWN_DIVERGENCES.md` — every cross-Python discrepancy observed (currently empty as of v0.0.5-alpha; verified no new divergences in Phase 6)
- [x] `NOT_YET_PORTED.md` — explicit list of out-of-scope components per §12
- [x] `examples/` — `load-thinc-model/`, `load-spacy-bundle/`, `tokenize/`, `bench/`
- [x] Tag `v3.8.14-port.0` (semver release, local-only — push is a maintainer-only step)
- [ ] Announcement post (blog / r/golang / spaCy discussions) — deferred; user-only action outside this plan
- [x] Update `UPSTREAM` file with final pinned commit and `our_release: v3.8.14-port.0`
- [x] `LICENSE` (MIT, matches upstream)
- [x] `CHANGELOG.md` — v0.1.0 / v3.8.14-port.0 entry

**Exit criteria**:
- `go get github.com/bioshock/gospacy/v3@v3.8.14-port.0` will work once the tag is pushed (tag created locally in Task 14; push is a maintainer follow-up).
- README quickstart compiles against the current exports (verified in Task 11 Step 3).
- All §5 compat targets met (100% on every measured attribute — see CHANGELOG v3.8.14-port.0).

**Notes**:
- 2026-05-19: Phase 6 complete. ~14 commits across docs + examples + godoc edits.
  New repo-root files: LICENSE, NOT_YET_PORTED.md. Two new examples: tokenize/, bench/.
  Stale Phase-N references in `doc/token.go`, `bundle/bundle.go`, and
  `pipeline/{tagger,parser,attributeruler_loader,lemmatizer}.go` cleaned up.
  Tagged `phase-6-complete` (internal) and `v3.8.14-port.0` (release, local-only).

---

## Phase 7 — Coverage expansion (helpers + perf + md/lg + NER)

Four-block plan (see `docs/superpowers/plans/2026-05-21-phase-7-coverage-expansion.md`).
Closes the gap between the v0.1 minimal-surface release and a usable
day-to-day port: workflow helpers, perf baseline, md/lg model support,
NER. Tagged `v3.8.14-port.2` at end of Block D.

### Block A — Merge workflow helpers (✅ 2026-05-21)
- [x] Rebase `differential/real-world-workflows` onto master (15 commits net; 2 fix commits dedup'd via patch-id)
- [x] FF merge into master
- [x] CHANGELOG + PROGRESS entries

### Block B — Performance investigation (✅ 2026-05-21)
- [x] Bench harness on real bundle (`bundle.Bundle.Pipe` throughput)
- [x] Identify hot path; document baseline in BENCHMARKS.md
- [x] One targeted optimisation: Lemmatizer per-POS cache → 2.2× speedup; Go now 2.2× faster than Python end-to-end

### Block C — `en_core_web_md` + `en_core_web_lg` (✅ 2026-05-21)
- [x] StaticVectors populated Forward (per-layer parity 1e-6)
- [x] MultiHashEmbed.v2 7-arm shape (include_static_vectors=true)
- [x] bundle.FromDisk md/lg vector matrix loading
- [x] md tagger + parser strict 100% (68/68 Tag/POS/Morph/Lemma/UAS/LAS)
- [x] lg tagger + parser strict 100% (68/68 across the board)

### Block D — NER + final tag (✅ 2026-05-21)
- [x] BiluoPushDown transition system + LoadBiluoMoves (74 moves sm/md)
- [x] State extended for NER (entOpen, entIOB, entities + methods)
- [x] registry.TransitionBasedParser.v2 wires state_type=ner
- [x] pipeline.NER greedy BILUO decode + Token.EntIOB/EntType writeback
- [x] Bundle.Pipe runs NER after lemmatizer (NER has its own non-listener tok2vec)
- [x] Per-case NER golden + ner_real_md test → strict 68/68 EntIOB + 68/68 EntType
- [x] Workflow W18 entity extraction (18/18 workflows green)
- [x] Tag `v3.8.14-port.2` (v0.2.0-alpha) — covers all four blocks

**Exit criteria**: All four blocks merged; v3.8.14-port.2 tagged; CHANGELOG cumulative entry covers helpers + perf + md/lg + NER. ✅ met.

**Notes**:
- 2026-05-21: All four blocks complete. Pipeline now `tokenize → tagger → parser → AR → lemmatizer → ner` for sm/md/lg. Differentials: Tag/POS/Morph/Lemma/UAS/LAS strict 100% on sm+md+lg (68/68 across 8 fixtures); NER EntIOB+EntType strict 100% on md (sm verified manually); 18/18 workflows. Lemmatizer cache (B4) drops `Bundle.Pipe` p50 from ~8 ms/op to ~3.67 ms/op (Go is 2.2× faster than Python). `v3.8.14-port.2` tagged locally at HEAD.

---

## Phase 8 — Sustained maintenance (ongoing)

- [ ] Quarterly upstream sync calendar reminder set (Q3 2026, Q4 2026, etc.)
- [ ] `docs/upstream-syncs/` directory created with first sync doc template
- [ ] `CONTRIBUTING.md` with good-first-issue labels for sync chores
- [ ] Issue templates for: bundle-load failure, divergence report, performance regression
- [ ] At least one external anchor user identified beyond the long-claim regression bench
- [ ] LTS policy documented (one prior lineage gets security-only patches for 6 months)

**Exit criteria**: ongoing, no terminal state. Annual review against §10 success criteria.

**Notes**:

---

## Cross-cutting tracking

These don't belong to a single phase but need to exist by v0.1:

- [x] `UPSTREAM` file kept current with every release (v3.8.14-port.0 as of 2026-05-19)
- [x] `KNOWN_DIVERGENCES.md` updated whenever a divergence is observed (currently empty — all Phase-4/4.5 divergences resolved in v0.0.5-alpha)
- [x] `NOT_YET_PORTED.md` exists as of v3.8.14-port.0; update whenever a user requests something out-of-scope
- [x] `BENCHMARKS.md` updated after every perf-affecting change (last update: Phase 2)
- [x] `CHANGELOG.md` updated per release (last entry: v3.8.14-port.0)

---

## Blockers / open questions

<!-- log blockers here with a date and short description; remove once resolved -->

_(none yet)_

---

End.
