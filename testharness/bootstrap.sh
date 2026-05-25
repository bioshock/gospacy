#!/usr/bin/env bash
# Creates the Python reference venv used by the differential test harness.
# Idempotent: rerun safely after `make clean`.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$HERE/requirements-ref.txt"

echo "ref env ready: $VENV"
echo "spaCy version:"
"$VENV/bin/python" -c "import spacy; print(spacy.__version__)"
