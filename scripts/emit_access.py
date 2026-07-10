"""Emit the real decode-token memory access stream from the packed manifest.

Walks every tensor in kernel read order (the order pack_weights laid them out)
and emits the byte ranges actually read for ONE decode token: all int8 payloads
+ their scales + the fp32 tensors. This is the true 758 MB/token access pattern,
not a synthetic one -- the whole point of packing the real weights first.

Output: artifacts/access.txt  -- "offset nbytes" per line, ascending offset.
The HBM harness turns each range into 64 B read transactions.
"""

import json
import sys

MANIFEST = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qwen35_int8.manifest.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "artifacts/access.txt"

m = json.load(open(MANIFEST))
ranges = []
for t in m["tensors"]:
    # int8 payload
    ranges.append((t["offset"], t["nbytes"]))
    # per-row fp32 scales sit right after an int8 tensor and are read too
    if t["dtype"] == "int8" and "scale_offset" in t:
        ranges.append((t["scale_offset"], t["scale_nbytes"]))

ranges.sort()
total = sum(n for _, n in ranges)
with open(OUT, "w") as f:
    f.write(f"# {len(ranges)} ranges, {total} bytes, from {MANIFEST}\n")
    for off, n in ranges:
        f.write(f"{off} {n}\n")

print(f"wrote {OUT}: {len(ranges)} ranges, {total/1e6:.1f} MB/token "
      f"(manifest total incl. padding {m['total_bytes']/1e6:.1f} MB)")
