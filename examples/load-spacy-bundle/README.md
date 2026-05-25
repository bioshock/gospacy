# Example: load-spacy-bundle

Loads a `.spacy` model directory and prints what gospacy understands of it:
language, pipeline, tokenizer output for a sample sentence, and per-pipe status.

## Running

Download a model (one-time, via the testharness):

```bash
make bootstrap-ref
make download-assets
```

Then build and run:

```bash
go run ./examples/load-spacy-bundle ./testdata/models/en_core_web_sm
```

Expected output (excerpt):

```
Language: en
Pipeline: [tok2vec tagger parser attribute_ruler lemmatizer ner]

Pipe("Hello world. Don't go to the U.S.A. today!"):
  IDX  TEXT            TAG    POS    MORPH                     LEMMA        HEAD DEP
  0    Hello           UH     INTJ                             hello        2    intj
  1    world           NN     NOUN   Number=Sing               world        0    npadvmod
  2    .               .      PUNCT  PunctType=Peri            .            0    punct
  ...
```

Pipes loaded: tok2vec, tagger, parser, attribute_ruler, lemmatizer.
Pipes skipped: senter (disabled in `nlp.disabled`), ner (out of scope — see
NOT_YET_PORTED.md).

## What this demonstrates

- gospacy reads a Python-trained `.spacy` bundle from disk.
- The tokenizer produces Python-matching token streams (verified by the 10k-corpus test).
- The full pipeline — tokenizer + tagger + parser + attribute_ruler + lemmatizer
  — runs end-to-end with 100% Tag/POS/Morph/Lemma/Head/Dep match on the 8
  fixture sentences. See `CHANGELOG.md` v0.0.5-alpha for the per-attribute
  breakdown.
- `ner` and `senter` remain skipped; `NOT_YET_PORTED.md` explains why.
