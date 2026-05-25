"""Run every workflow dump script in this directory.

Iterates dump_w*.py in lex order, invoking each script's main(). Cheap to
re-run end-to-end (8 cases × ~17 workflows). Used by `make goldens-workflows`.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    scripts = sorted(p for p in HERE.glob("dump_w*.py"))
    if not scripts:
        print("no dump_w*.py scripts found", file=sys.stderr)
        return 1
    for script in scripts:
        mod_name = script.stem
        spec = importlib.util.spec_from_file_location(mod_name, script)
        if spec is None or spec.loader is None:
            print(f"could not load {script}", file=sys.stderr)
            return 1
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.main()
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
