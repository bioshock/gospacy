#!/usr/bin/env python3
"""Dump structural metadata for en_core_web_sm.

Output: testdata/golden/bundle_meta.json
Schema: {"lang": str, "pipeline": [str], "components": {name: {"architecture": str|null}}}
"""

import json
import sys
from pathlib import Path

import spacy
from spacy.util import load_config

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"
MODEL = REPO / "testdata" / "models" / "en_core_web_sm"


def main() -> int:
    if not MODEL.exists():
        print(f"missing model: {MODEL}", file=sys.stderr)
        return 1
    cfg = load_config(MODEL / "config.cfg")
    pipeline = list(cfg["nlp"]["pipeline"])
    components = {}
    for name in cfg.get("components", {}):
        comp = cfg["components"][name]
        arch = None
        if isinstance(comp.get("model"), dict):
            arch = comp["model"].get("@architectures")
        components[name] = {"architecture": arch}
    out = {
        "lang": cfg["nlp"]["lang"],
        "pipeline": pipeline,
        "components": components,
    }
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / "bundle_meta.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
