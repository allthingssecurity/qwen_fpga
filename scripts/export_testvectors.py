"""Export golden delta-rule vectors for the HLS csim testbench.

Captures the state at a *warm* step (state non-trivial, gates non-unity), which
is where a hardware delta rule actually breaks -- a cold zero state hides sign
errors in the rank-1 update.

  artifacts/tv_deltanet.bin  : q,k,v,g,beta,S_in,o_core,S_out  little-endian fp32
  artifacts/tv_deltanet.json : shapes + offsets
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
import model as M  # noqa: E402
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import DecodeState, decode_step  # noqa: E402

LAYER = 0      # linear-attn index (not absolute layer index)
WARM_STEPS = 5


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap)

    # warm the state up so S is dense and the gates are away from 1.0
    st = DecodeState.new(cfg)
    toks = [760, 6511, 314, 9338, 369]
    for t in toks[:WARM_STEPS]:
        decode_step(W, cfg, t, st)

    M.TRACE = {"layer": LAYER}
    decode_step(W, cfg, 11751, st)
    tr = M.TRACE
    M.TRACE = None

    for k in ("q", "k", "v", "g", "beta", "S_in", "o_core", "S_out"):
        assert k in tr, f"trace missing {k}"

    gexp = np.exp(tr["g"])
    print(f"captured linear layer {LAYER} at step {WARM_STEPS}")
    print(f"  |S_in|  mean {np.abs(tr['S_in']).mean():.4e}  max {np.abs(tr['S_in']).max():.4e}")
    print(f"  exp(g)  min {gexp.min():.4f}  max {gexp.max():.4f}   (1.0 would mean no decay)")
    print(f"  beta    min {tr['beta'].min():.4f}  max {tr['beta'].max():.4f}")
    print(f"  |o_core| max {np.abs(tr['o_core']).max():.4e}")
    assert np.abs(tr["S_in"]).max() > 1e-3, "state is cold -- warm it more"

    os.makedirs("artifacts", exist_ok=True)
    meta, off = [], 0
    with open("artifacts/tv_deltanet.bin", "wb") as f:
        for name in ("q", "k", "v", "g", "beta", "S_in", "o_core", "S_out"):
            a = np.ascontiguousarray(tr[name], dtype="<f4")
            f.write(a.tobytes())
            meta.append({"name": name, "shape": list(a.shape), "offset": off, "nbytes": a.nbytes})
            off += a.nbytes

    with open("artifacts/tv_deltanet.json", "w") as f:
        json.dump({"H": cfg["linear_num_value_heads"], "K": cfg["linear_key_head_dim"],
                   "V": cfg["linear_value_head_dim"], "total_bytes": off, "tensors": meta}, f, indent=1)
    print(f"wrote artifacts/tv_deltanet.bin ({off/1024:.1f} KB) + .json")


if __name__ == "__main__":
    main()
