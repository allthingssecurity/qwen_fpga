"""Emit a flat weight index the C++ testbench parses to build WeightBlob.

One line per tensor:  name off nbytes dtype(0fp32/1int8) scale_off scale_nbytes out in
`out`/`in` are the 2-D GEMV dims (0 for non-matrix tensors).
"""

import json
import sys

MANIFEST = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qwen35_int8.manifest.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "artifacts/weights.idx"

m = json.load(open(MANIFEST))
with open(OUT, "w") as f:
    for t in m["tensors"]:
        sh = t["shape"]
        out_dim = sh[0] if len(sh) >= 1 else 0
        in_dim = sh[1] if len(sh) >= 2 else 0
        dtype = 1 if t["dtype"] == "int8" else 0
        so = t.get("scale_offset", 0)
        sn = t.get("scale_nbytes", 0)
        f.write(f"{t['name']} {t['offset']} {t['nbytes']} {dtype} {so} {sn} {out_dim} {in_dim}\n")
print(f"wrote {OUT}: {len(m['tensors'])} tensors")
