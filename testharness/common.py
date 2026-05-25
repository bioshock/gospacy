"""Shared helpers for differential-test dump scripts."""

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

import spacy
from spacy.language import Language

REPO = Path(__file__).resolve().parent.parent
# GOSPACY_MODEL selects which bundle to load (defaults to sm for backward
# compatibility with the existing per-bundle goldens). Used by Block C's md/lg
# dumpers to materialise parallel goldens against the same fixture corpus.
MODEL_NAME = os.environ.get("GOSPACY_MODEL", "en_core_web_sm")
MODEL_PATH = REPO / "testdata" / "models" / MODEL_NAME
GOLDEN = REPO / "testdata" / "golden"
CORPORA = REPO / "testdata" / "corpora"


def golden_suffix() -> str:
    """Returns the filename suffix to use for goldens of the active model.

    sm → "" (existing files untouched); md → "_md"; lg → "_lg".
    """
    if MODEL_NAME == "en_core_web_sm":
        return ""
    # en_core_web_md → _md ; en_core_web_lg → _lg
    return "_" + MODEL_NAME.split("_")[-1]


def load_nlp() -> Language:
    if not MODEL_PATH.exists():
        print(f"missing model: {MODEL_PATH}; run `make download-assets`", file=sys.stderr)
        sys.exit(1)
    return spacy.load(str(MODEL_PATH))


def iter_corpus_lines(path: Path) -> Iterator[str]:
    """Yield one sentence per line, stripping trailing newline. Skips blanks."""
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line:
                yield line


def write_json(path: Path, payload) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"wrote {path}")
