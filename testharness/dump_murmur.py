#!/usr/bin/env python3
"""Dump Python murmurhash reference vectors for cross-language verification.

Output: testdata/golden/murmur_vectors.json

Schema:
{
  "hash64": [
    {"key": "...", "seed": 0, "value_hex": "0x...", "value_dec": "..."},
    ...
  ],
  "hash3_x86_128_uint64": [
    {"key": 42, "seed": 0, "value": [u0, u1, u2, u3]},
    ...
  ]
}
"""

import ctypes
import json
import sys
from pathlib import Path

import murmurhash  # noqa: F401 – verifies package is installed

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "testdata" / "golden"


# String keys for hash64 (spaCy StringStore usage).
HASH64_CASES = [
    ("", 0),
    ("", 1),
    ("a", 0),
    ("a", 1),
    ("hello", 0),
    ("hello", 1),
    ("the", 1),
    ("spacy", 1),
    ("The quick brown fox jumps over the lazy dog.", 1),
    ("ünïcødé", 1),
    ("a" * 15, 1),  # exactly fills 15-byte tail
    ("a" * 16, 1),  # exactly fills one 16-byte block
    ("a" * 17, 1),  # one block + 1-byte tail
    ("a" * 32, 1),  # two blocks
]

# uint64 keys for hash3_x86_128_uint64 (thinc HashEmbed usage).
HASH128_CASES = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (42, 0),
    (42, 1),
    (0xFFFFFFFFFFFFFFFF, 1),
    (123456789, 7),
]


def _load_hash64():
    """Load MurmurHash64A from the murmurhash .so via ctypes.

    murmurhash.mrmr exposes hash64 only as a C-level cdef, so we reach it
    directly from the compiled extension.
    """
    import murmurhash.mrmr as mrmr
    so_path = mrmr.__file__
    lib = ctypes.CDLL(so_path)
    # C++ mangled name on Linux x86-64:
    #   MurmurHash64A(void const*, int, unsigned long) -> unsigned long
    fn = lib._Z13MurmurHash64APKvim
    fn.restype = ctypes.c_uint64
    fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_ulong]
    return fn


def main() -> int:
    _hash64 = _load_hash64()
    out = {"hash64": [], "hash3_x86_128_uint64": []}

    for key, seed in HASH64_CASES:
        encoded = key.encode("utf-8")
        v = _hash64(encoded, len(encoded), seed)
        out["hash64"].append({
            "key": key,
            "seed": seed,
            "value_hex": f"0x{v & 0xFFFFFFFFFFFFFFFF:016x}",
            "value_dec": str(v & 0xFFFFFFFFFFFFFFFF),
        })

    # murmurhash exposes a low-level API but not directly the x86_128_uint64
    # used by thinc. We invoke thinc's path instead so the test vectors match
    # exactly what HashEmbed sees.
    import numpy as np
    from thinc.api import NumpyOps
    ops = NumpyOps()
    for key, seed in HASH128_CASES:
        ids = np.array([key], dtype=np.uint64)
        result = ops.hash(ids, seed)  # shape (1, 4) uint32
        out["hash3_x86_128_uint64"].append({
            "key": int(key),
            "seed": int(seed),
            "value": [int(x) for x in result[0]],
        })

    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / "murmur_vectors.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
