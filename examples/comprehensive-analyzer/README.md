# comprehensive-analyzer

Go port of a "comprehensive spaCy" demo — exercises tokens / NER /
noun chunks / dependencies / sentences / rule-based matching / vector
similarity in one program. Equivalent to the canonical
`ComprehensiveSpacyAnalyzer` Python script.

## Usage

```bash
go run ./examples/comprehensive-analyzer /path/to/en_core_web_md
```

You need a `_md` or `_lg` bundle for the similarity section to be
meaningful — `_sm` has empty vectors. Download via spaCy:

```bash
testharness/.venv/bin/python -m spacy download en_core_web_md
# then place under testdata/models/en_core_web_md/ (see
# testharness/download_assets.sh for the layout the bundle loader
# expects).
```

## What the example covers

| # | Section | gospacy equivalent |
|---|---|---|
| 1 | Token-level (text / lemma / POS / Tag / IS_ALPHA / IS_STOP) | `doc.Tokens` + `StringStore.LookupOrEmpty` + `Token.IsStop(vocab)` |
| 2 | Named entities | hand-rolled walk over `Tokens[].EntIOB` (B-/I-/O scheme) |
| 3 | Noun chunks | `Doc.NounChunks()` |
| 4 | Dependency parsing | `Token.Dep` / `Token.Head` |
| 5 | Sentence segmentation | `Doc.Sents()` (reads parser-set `SentStart`) |
| 6 | Rule-based matching | `matcher.New(vocab) + m.Add("KEY", []TokenSpec{...})` |
| 7 | Document similarity | mean-pooled cosine over `vocab.Vectors().Row(tok.Lower)` |

## Two places it diverges from Python spaCy

**`spacy.matcher.Matcher` ships as Tier 1 only** — equality, set
(`IN`/`NOT_IN`), and `REGEX` on LOWER. Quantifier OPs (`?` / `*` / `+`
/ `!` / `{n,m}`), `PhraseMatcher`, and `FUZZY` are NOT_YET_PORTED. For
the AI / Artificial-Intelligence pattern Tier 1 is plenty — the
example expresses "optional Intelligence" as two alternatives under
the same key, and same-key overlap dedup (longest-first) picks the
longer match when both fire.

**`Doc.similarity` isn't a method.** spaCy's default
`doc1.similarity(doc2)` is "mean of per-token vectors, cosine". The
example computes that inline using `vocab.Vectors().Row(hash)`, which
gives you the 300-dim static vector for any in-vocab lemma. OOV tokens
are skipped (matching spaCy). Add it as a Doc method in your codebase
if you do this often.

**`spacy.explain` is a small table.** gospacy doesn't ship the
labels-to-descriptions map. The example inlines the OntoNotes NER
subset (PERSON, ORG, GPE, MONEY, DATE, ...). Extend the `explain()`
function if you need POS or dep-label descriptions too.

## Output (abridged, on the sample text)

```
Loading spaCy model from: testdata/models/en_core_web_md

--- 1. Token Level Analysis ---
TEXT            | LEMMA           | POS        | TAG        | IS_ALPHA   | IS_STOP
---------------------------------------------------------------------------
Apple           | Apple           | PROPN      | NNP        | true       | false
Inc.            | Inc.            | PROPN      | NNP        | false      | false
is              | be              | AUX        | VBZ        | true       | true
considering     | consider        | VERB       | VBG        | true       | false
...

--- 2. Named Entity Recognition (NER) ---
Entity: Apple Inc.                | Label: ORG        | Explanation: Companies, agencies, institutions, etc.
Entity: U.K.                      | Label: GPE        | Explanation: Countries, cities, states
Entity: $1 billion                | Label: MONEY      | Explanation: Monetary values, including unit
Entity: Tim Cook                  | Label: PERSON     | Explanation: People, including fictional
...

--- 6. Rule-based Matching ---
Match Rule: AI_PATTERN   | Matched Text: "Artificial Intelligence"
Match Rule: AI_PATTERN   | Matched Text: "AI"

--- 7. Document Similarity ---
Text 1: Apple Inc. is considering buying a U.K. startup for $1 billion. ...
Text 2: A British technology company might be purchased by Apple for one billion dollars.
Similarity Score: 0.8xxx
```
