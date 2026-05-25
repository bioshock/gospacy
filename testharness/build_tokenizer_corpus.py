#!/usr/bin/env python3
"""
Builds testdata/corpora/tokenizer-10k.txt — a deterministic 10k-sentence corpus
for tokenizer differential testing, sampled from UD English-EWT.

Output: one raw sentence per line, no annotation. Reproducible: same UD-EWT
input → same output.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UD = REPO / "testdata" / "corpora" / "ud-english-ewt"
OUT = REPO / "testdata" / "corpora" / "tokenizer-10k.txt"

TARGET = 10_000


def iter_sentences(conllu_path: Path):
    """Yield raw sentences from a CoNLL-U file (# text = ... lines)."""
    with conllu_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# text = "):
                yield line[len("# text = "):]


def main() -> int:
    if not UD.exists():
        print(f"missing {UD}; run `make download-assets` first", file=sys.stderr)
        return 1

    sentences = []
    for split in ("en_ewt-ud-train.conllu", "en_ewt-ud-dev.conllu", "en_ewt-ud-test.conllu"):
        path = UD / split
        if not path.exists():
            print(f"missing split: {path}", file=sys.stderr)
            return 1
        for s in iter_sentences(path):
            sentences.append(s)
            if len(sentences) >= TARGET:
                break
        if len(sentences) >= TARGET:
            break

    if len(sentences) < TARGET:
        print(f"only got {len(sentences)} sentences; UD-EWT may have shrunk", file=sys.stderr)

    OUT.write_text("\n".join(sentences[:TARGET]) + "\n", encoding="utf-8")
    print(f"wrote {min(len(sentences), TARGET)} sentences to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
