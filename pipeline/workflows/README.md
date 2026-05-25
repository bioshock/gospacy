# pipeline/workflows

This package validates gospacy's public API by porting 17 real-world spaCy
workflows from spaCy 101 docs and Stack Overflow patterns; each ships with a
Python golden generator and a strict 100% Go-vs-Python differential.

Each workflow is a pure function `func(d *doc.Doc, ss *vocab.StringStore) any`
that returns a JSON-marshalable value. The test harness pipes 8 fixture
sentences once through `Bundle.Pipe`, runs every registered workflow, and
asserts that the canonical-JSON output equals the matching Python golden under
`testdata/golden/workflows/`.

## API surface

The workflows are built on these helpers, added on this branch:

```go
// doc/sents.go — sentence iterator over parser-set SentStart markers.
func (d *Doc) Sents() []Span

// doc/children.go — dependency-tree navigation.
func ChildrenOf(d *Doc, self int) []int
func SubtreeOf(d *Doc, self int) []int

// doc/noun_chunks.go — port of spacy.lang.en.syntax_iterators.noun_chunks.
func (d *Doc) NounChunks() []Span

// doc/token_stop.go — English stop-word check.
func (t Token) IsStop(v *vocab.Vocab) bool

// vocab/stopwords_en.go — generated 326-entry English STOP_WORDS set.
func IsStopEN(lower string) bool
```

**API divergence from the plan**: `Children` / `Subtree` ship as
package-level functions on `doc` (`doc.ChildrenOf(d, i)` /
`doc.SubtreeOf(d, i)`) rather than as `Token` methods. `Token` is a value
type with no embedded index field, so it has no way to know its own position
in `Doc.Tokens`; the caller must pass both the owning `*Doc` and the index.

The English stop-word set is regenerated from upstream Python by
`internal/cmd/genstopwords`.

## Workflow catalogue

| # | Name | What it computes | Source | Fields | Helper | Golden |
|---|---|---|---|---|---|---|
| W01 | `w01_pos_freq` | POS-tag frequency map over the Doc | spaCy 101 §POS tagging | `POS` | — | `testdata/golden/workflows/w01_pos_freq.json` |
| W02 | `w02_propn` | Surface text of every `POS == PROPN` token | spaCy 101 §Named entities (PROPN filter) | `POS`, `Text` | — | `testdata/golden/workflows/w02_propn.json` |
| W03 | `w03_comparatives` | `{text, tag}` for every JJR/JJS/RBR/RBS token | SO question pattern (comparatives) | `Tag`, `Text` | — | `testdata/golden/workflows/w03_comparatives.json` |
| W04 | `w04_unique_lemmas` | Sorted distinct lemmas | spaCy 101 §Lemmatization | `Lemma` | — | `testdata/golden/workflows/w04_unique_lemmas.json` |
| W05 | `w05_content_words` | Lemmas of non-stop, non-punct tokens | SO question pattern (content-word filter) | `Lemma`, `Lower`, `Orth` | `Token.IsStop`, `lex.IsPunct` | `testdata/golden/workflows/w05_content_words.json` |
| W06 | `w06_lemma_freq` | Lemma freq excluding stop words + punct | spaCy 101 §Vocab, lemmas | `Lemma`, `Lower`, `Orth` | `Token.IsStop`, `lex.IsPunct` | `testdata/golden/workflows/w06_lemma_freq.json` |
| W07 | `w07_sent_roots` | `{text, dep, pos}` for every `dep_ == "ROOT"` | spaCy 101 §Dependency parsing | `Dep`, `POS`, `Text` | — | `testdata/golden/workflows/w07_sent_roots.json` |
| W08 | `w08_svo` | `[subj, verb, obj]` triples for VERBs with nsubj+dobj children | SO question pattern (subject-verb-object) | `POS`, `Dep`, `Text` | `doc.ChildrenOf` | `testdata/golden/workflows/w08_svo.json` |
| W09 | `w09_passive` | `{text, head}` for every `dep_ == "nsubjpass"` | spaCy 101 §Dependency parsing | `Dep`, `Head`, `Text` | — | `testdata/golden/workflows/w09_passive.json` |
| W10 | `w10_negation` | `{neg, head, head_pos}` for every `dep_ == "neg"` | SO question pattern (negation extraction) | `Dep`, `Head`, `POS`, `Text` | — | `testdata/golden/workflows/w10_negation.json` |
| W11 | `w11_sent_segmentation` | Per-sentence surface text | spaCy 101 §Sentence segmentation | `SentStart`, token text | `Doc.Sents` | `testdata/golden/workflows/w11_sent_segmentation.json` |
| W12 | `w12_sent_count` | Number of sentences in the Doc | spaCy 101 §Sentence segmentation | `SentStart` | `Doc.Sents` | `testdata/golden/workflows/w12_sent_count.json` |
| W13 | `w13_plural_nouns` | NOUN tokens with `Number=Plur` morphology | SO question pattern (plural extraction) | `POS`, `Morph`, `Text` | — | `testdata/golden/workflows/w13_plural_nouns.json` |
| W14 | `w14_past_tense` | VERB tokens with `Tense=Past` morphology | SO question pattern (tense filter) | `POS`, `Morph`, `Text` | — | `testdata/golden/workflows/w14_past_tense.json` |
| W15 | `w15_noun_chunks` | `{start, end, text}` for each base noun phrase | spaCy 101 §Noun chunks | `POS`, `Dep`, `Head` | `Doc.NounChunks` | `testdata/golden/workflows/w15_noun_chunks.json` |
| W16 | `w16_questions` | One bool per sentence — first tag starts `W*` OR last token is `"?"` | SO question pattern (question detection) | `SentStart`, `Tag`, `Text` | `Doc.Sents` | `testdata/golden/workflows/w16_questions.json` |
| W17 | `w17_keywords` | NOUN/PROPN lemma frequency excluding stop words | SO question pattern (keyword extraction) | `POS`, `Lemma`, `Lower` | `Token.IsStop` | `testdata/golden/workflows/w17_keywords.json` |

The 8 fixture sentences are the same `testharness/pipeline_cases.json`
used by the rest of the differential suite — workflows piggyback on
tagger/parser/AR/lemma fields that already match Python 100% per token.

## Regenerating goldens

```bash
testharness/.venv/bin/python testharness/workflows/dump_all.py
```

This walks every `dump_w*.py` script in `testharness/workflows/` in lex order
and rewrites the corresponding `testdata/golden/workflows/wNN_*.json`. The
dumps are deterministic — re-running yields byte-identical output.

There is no dedicated `make` target; the umbrella above is the canonical
entry point.

## Running the differential

```bash
# All 17 workflows.
go test ./pipeline/workflows -run TestWorkflows_RealBundle -v

# A single workflow (Go subtest name == workflow Name).
go test ./pipeline/workflows -run TestWorkflows_RealBundle/w08_svo -v
```

Strict 100% equality is asserted via canonical JSON (Go map keys sorted,
arrays preserved in source order). Any divergence indicates either helper
drift (`Sents` / `Children` / `NounChunks` / `IsStop`) or a real port gap
in the underlying pipeline — never tagger/parser noise, since v0.1 already
matches Python 100% on POS / Tag / Morph / Lemma / Dep / Head per token.

## Failure-mode protocol

If a workflow fails:

1. **Isolate** — re-run the single workflow with `-run
   TestWorkflows_RealBundle/wNN_<name> -v` and dump both sides
   (`got` vs Python golden).
2. **Classify** — does the diff trace to a helper (`Sents` / `ChildrenOf` /
   `NounChunks` / `IsStop`), or to an upstream pipeline field
   (POS/Dep/Morph/Lemma)? Helper drift is fixed in `doc/`; pipeline drift
   is fixed upstream (parser/tagger/AR) and the workflow is retested.
3. **Document** — record the divergence in `KNOWN_DIVERGENCES.md` (with
   the failing case + root cause) if a fix is deferred, or drop the
   workflow from the differential and append a note to
   `NOT_YET_PORTED.md`. Silent skips are forbidden — see the project's
   Rule 12 ("Fail loud").

This protocol is the canonical reference; the original branch plan in
`docs/` references the same three-step sequence.
