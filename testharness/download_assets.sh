#!/usr/bin/env bash
# Fetches reference model + corpora into testdata/.
# These directories are gitignored; commit only the small JSON goldens.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/testharness/.venv"
MODELS="$REPO_ROOT/testdata/models"
CORPORA="$REPO_ROOT/testdata/corpora"

mkdir -p "$MODELS" "$CORPORA"

# 1. Reference model: en_core_web_sm (compatible with pinned spaCy)
if [ ! -d "$MODELS/en_core_web_sm" ]; then
  "$VENV/bin/python" -m spacy download en_core_web_sm
  # spacy installs into the venv's site-packages; copy to testdata for path stability.
  MODEL_SRC="$("$VENV/bin/python" -c 'import en_core_web_sm; import os; print(os.path.dirname(en_core_web_sm.__file__))')"
  # The model dir is one level down; find it
  VERSIONED="$(ls -d "$MODEL_SRC"/en_core_web_sm-* | head -1)"
  cp -r "$VERSIONED" "$MODELS/en_core_web_sm"
fi

# 1b. Medium model: en_core_web_md (300-dim static vectors, ~40 MB)
if [ ! -d "$MODELS/en_core_web_md" ]; then
  "$VENV/bin/python" -m spacy download en_core_web_md
  MODEL_SRC_MD="$("$VENV/bin/python" -c 'import en_core_web_md; import os; print(os.path.dirname(en_core_web_md.__file__))')"
  VERSIONED_MD="$(ls -d "$MODEL_SRC_MD"/en_core_web_md-* | head -1)"
  cp -r "$VERSIONED_MD" "$MODELS/en_core_web_md"
fi

# 1c. Large model: en_core_web_lg (685k vectors, ~560 MB)
if [ ! -d "$MODELS/en_core_web_lg" ]; then
  "$VENV/bin/python" -m spacy download en_core_web_lg
  MODEL_SRC_LG="$("$VENV/bin/python" -c 'import en_core_web_lg; import os; print(os.path.dirname(en_core_web_lg.__file__))')"
  VERSIONED_LG="$(ls -d "$MODEL_SRC_LG"/en_core_web_lg-* | head -1)"
  cp -r "$VERSIONED_LG" "$MODELS/en_core_web_lg"
fi

# 2. UD English-EWT — treebank with tokens, POS, lemmas, arcs
UD_DIR="$CORPORA/ud-english-ewt"
if [ ! -d "$UD_DIR" ]; then
  git clone --depth=1 https://github.com/UniversalDependencies/UD_English-EWT.git "$UD_DIR"
fi

echo "Assets ready:"
du -sh "$MODELS"/* "$CORPORA"/*
