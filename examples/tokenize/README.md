# Example: tokenize

The smallest gospacy demo. Builds the English tokenizer rules, runs them on a
string, prints one row per token. No model bundle required — pure rule-based
tokenization.

## Running

```bash
go run ./examples/tokenize
go run ./examples/tokenize "Mr. Smith went to Washington."
```

## What this demonstrates

- The tokenizer is independent of any `.spacy` bundle. You can tokenize without
  loading a 14 MB model.
- The `tokenizer.Tokenizer` / `vocab.Vocab` / `lang/en.MakeRules` triple is the
  minimum useful API surface.
- Token streams are byte-for-byte identical to Python's `spacy.lang.en.English()`
  tokenizer — verified by the 10k-sentence corpus differential test.

For tagger / parser / lemmatizer output, see `examples/load-spacy-bundle/`.
