# Example: load-thinc-model

A minimal CLI that demonstrates the gospacy library's public API:
- Build a `nn.Model` tree using `layers.Chain`, `layers.Linear`, `layers.Softmax`.
- Load weights from a Python-trained thinc `.msgpack` file via `model.FromBytes`.
- Run a forward pass via `model.Predict`.

## Running

First, produce the model file (from the repo root):

```bash
make bootstrap-ref      # one-time: install Python ref env
make download-assets    # one-time: fetch model
testharness/.venv/bin/python testharness/dump_thinc_model.py
```

Then build and run the example:

```bash
go run ./examples/load-thinc-model ./testdata/golden/tiny_thinc_model.msgpack
```

Expected output (3 rows of softmax-normalised 2-class predictions, each summing to ~1.0):

```
Output shape: (3, 2)
  row 0: [<float> <float>]
  row 1: [<float> <float>]
  row 2: [<float> <float>]
```

## What this proves

- gospacy can load a model trained in Python and predict with no Python runtime.
- The public API surface is: `nn`, `nn/backend/gonum`, `nn/layers`. That's it.
- For larger models, build the matching Go tree (one Linear/Softmax/Maxout/etc. per layer in the Python tree, in pre-order); `FromBytes` populates the weights.

For a non-trivial model (e.g., loading `en_core_web_sm`), see
`examples/load-spacy-bundle/`, which uses the `bundle.FromDisk` entry point that
wraps the architecture registry, tokenizer, and per-pipe weight loading. This
example is the minimum viable demonstration of the `nn` package itself.
