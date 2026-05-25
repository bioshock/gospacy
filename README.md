# gospacy

A Go port of [Explosion's spaCy](https://github.com/explosion/spaCy), inference only.

**Status:** Alpha — `v3.8.14-port.0`. Phases 0 through 5 of the [roadmap](OVERVIEW.md#6-phased-roadmap) are shipped: tokenizer, tagger, dependency parser, attribute_ruler, and lemmatizer all run end-to-end against `en_core_web_sm`. Pure-Go, CPU, English-only — see [Limitations](#limitations) below.

**Goal:** `go get` a Go library that loads `.spacy` model bundles trained in Python and runs the inference pipeline with no Python runtime required.

---

## Quickstart

```bash
go get github.com/bioshock/gospacy/v3@v3.8.14-port.0
```

```go
package main

import (
    "fmt"

    "github.com/bioshock/gospacy/v3/bundle"
)

func main() {
    b, err := bundle.FromDisk("./en_core_web_sm")
    if err != nil {
        panic(err)
    }
    doc, err := b.Pipe("Don't go to the U.S.A. today.")
    if err != nil {
        panic(err)
    }
    ss := b.Vocab.StringStore()
    for i := 0; i < doc.NumTokens(); i++ {
        tok := doc.Tokens[i]
        tag, _ := ss.Lookup(tok.Tag)
        pos, _ := ss.Lookup(tok.POS)
        dep, _ := ss.Lookup(tok.Dep)
        fmt.Printf("%-10s %-6s %-6s head=%-2d %s\n", tok.Text, tag, pos, tok.Head, dep)
    }
}
```

Get the model: download `en_core_web_sm-3.8.0.tar.gz` from
[github.com/explosion/spacy-models](https://github.com/explosion/spacy-models/releases?q=en_core_web_sm)
and extract it. The directory you pass to `bundle.FromDisk` is the one
containing `meta.json` and `config.cfg`.

---

## Supported models

v0.2 supports three English models — `sm`, `md`, `lg` — all verified at
strict 100% per-token Tag/POS/Morph/Lemma/Head/Dep parity vs Python on the
8-fixture corpus:

| Model | Version | Bundle architectures verified |
|---|---|---|
| `en_core_web_sm` | 3.8.x | tok2vec (`Tok2Vec.v2` / `MultiHashEmbed.v2` / `MaxoutWindowEncoder.v2`), tagger (`Tagger.v2`), parser (`TransitionBasedParser.v2` / `TransitionModel.v1` / `PrecomputableAffine.v1`), attribute_ruler, lemmatizer (rule + lookup modes) |
| `en_core_web_md` | 3.8.x | All sm architectures + `StaticVectors.v2` 7th-arm in `MultiHashEmbed.v2` (vocab/vectors 20000×300, key2row 684830 entries) |
| `en_core_web_lg` | 3.8.x | Same architecture as md, larger vector table (vocab/vectors 342918×300, same key2row) |

Other v3.x English models (`trf` transformer-based) and non-English models
are out of scope for v0.2 — see [`NOT_YET_PORTED.md`](NOT_YET_PORTED.md).

---

## Compatibility against Python spaCy 3.8.14

Measured end-to-end on 8 fixture sentences (68 tokens) from
`testharness/pipeline_cases.json`:

| Attribute | Match | Target ([OVERVIEW §5](OVERVIEW.md#5-compatibility-goals)) |
|---|---|---|
| Tag (fine-grained POS) | 68/68 (100%) | ≥99% pure-Go |
| POS (coarse Universal) | 68/68 (100%) | ≥99% pure-Go |
| Morphology | 68/68 (100%) | — (not in §5 table) |
| Lemma | 68/68 (100%) | — (not in §5 table) |
| Head (UAS) | 68/68 (100%) | ≥97% pure-Go |
| Dep label | 68/68 (100%) | ≥97% pure-Go |

These are the numbers from [`CHANGELOG.md`](CHANGELOG.md) v0.0.5-alpha. We
exceed every §5 target. If you observe a divergence on a larger corpus, please
file an issue; the runtime KNOWN_DIVERGENCES list is currently empty.

---

## Building

```bash
make build           # pure-Go default (no system deps)
make test            # pure-Go tests (~16 packages)
make test-blis       # cgo+BLIS tests — see Build flags below
```

### Build flags

- **Default (no tag):** pure-Go. Uses `gonum.org/v1/gonum` for gemm. No system
  dependencies. This is what `go get` produces.
- **`-tags blis`:** routes hot ops through a cgo wrapper around
  [libblis](https://github.com/flame/blis). Currently a scaffold — the cgo
  bindings are stubs. Not exercised by v0.1 differential tests. See
  [`NOT_YET_PORTED.md`](NOT_YET_PORTED.md#full-cgo--blis-bindings).

---

## Verifying against Python

```bash
make bootstrap-ref      # one-time: create Python ref venv
make download-assets    # one-time: fetch reference model and corpora
make diff-test          # regenerate golden fixtures from Python
make test               # Go tests load goldens and compare
```

---

## Examples

- [`examples/tokenize/`](examples/tokenize/) — smallest possible demo:
  English tokenizer only, no bundle.
- [`examples/load-spacy-bundle/`](examples/load-spacy-bundle/) — load a bundle
  and run the full pipeline on a sentence. Prints text/tag/pos/morph/lemma/head/dep.
- [`examples/load-thinc-model/`](examples/load-thinc-model/) — load a Python-trained
  thinc Chain model and run forward without any of the spaCy pipeline.
- [`examples/bench/`](examples/bench/) — throughput smoke test
  (sentences/sec, tokens/sec) on 100 short sentences.

---

## Project layout

```
nn/                — Ops + Model tree + layers + msgpack loader (the thinc port)
vocab/             — StringStore + Lexeme + Vocab + Vectors
tokenizer/         — Rule-based tokenizer (regexp2)
lang/en/           — English tokenizer rules and exceptions
config/            — config.cfg parser (INI-ish)
registry/          — 22 spaCy architecture factories
bundle/            — .spacy bundle reader (FromDisk + Pipe)
doc/               — Doc / Token / Span runtime types
pipeline/          — Tagger / Parser / AttributeRuler / Lemmatizer
internal/lookups/  — msgpack lookup table loader
testharness/       — Python ref scripts that produce goldens
```

See [`OVERVIEW.md`](OVERVIEW.md) for the full architecture document, the
upstream-pinning model, and the project roadmap.

---

## Versioning

Releases follow `vX.Y.Z-port.N` where `X.Y.Z` is the pinned upstream spaCy
version. `v3.8.14-port.0` ports spaCy 3.8.14, first iteration of our port. See
the [`UPSTREAM`](UPSTREAM) file at repo root for the pinned commit and
[OVERVIEW §3](OVERVIEW.md#3-versioning-model) for the full model.

---

## Limitations

gospacy v0.1 is intentionally narrow. The following are explicitly out of scope
and have a one-line rationale in [`NOT_YET_PORTED.md`](NOT_YET_PORTED.md):

- **Entity Linker.** Downstream of NER (which is ported in v0.2); no anchor demand yet.
- **Training, oracle, gradients.** Inference only.
- **Beam search.** Greedy decoding only (`beam_width=1`).
- **`senter`.** Disabled by default in `en_core_web_sm`; sentence boundaries
  come from the parser.
- **Languages other than English.** `nlp.lang != "en"` is a load error.
- **GPU.** CPU only.
- **Full cgo+BLIS.** Scaffold exists; bindings deferred (pure-Go is already
  2.2× faster than Python end-to-end as of v0.2).
- **Custom user pipeline components.** The pipe sequence is fixed
  (`tokenize → tagger → parser → AR → lemmatizer → ner`).

Runtime divergences from Python (when any exist) are tracked in
[`KNOWN_DIVERGENCES.md`](KNOWN_DIVERGENCES.md). The list is currently empty.

---

## License

MIT — see [`LICENSE`](LICENSE). Matches upstream spaCy and thinc.
