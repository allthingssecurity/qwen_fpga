"""Export the DeltaNet gated-norm case for head 0:
   on = o * rsqrt(mean(o^2) + eps)      (RMS normalize, mean-based)
   y  = norm_weight * on * silu(z)       (scale by w, gate by silu(z))
o is the delta-rule output (from tv_deltanet), z is the in_proj_z slice for the
same warm token. Also exports the sigmoid LUT for silu.

  python3 scripts/export_gnorm_tv.py  ->  artifacts/tv_gnorm.bin
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import rmsnorm, sigmoid, silu  # noqa: E402

V = 128
LUTN = 1024


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)
    p = "layers.0.linear_attn"

    # o (head 0) from the deltanet golden vectors
    meta = json.load(open("artifacts/tv_deltanet.json"))
    off = next(t["offset"] for t in meta["tensors"] if t["name"] == "o_core")
    raw = np.fromfile("artifacts/tv_deltanet.bin", dtype="<f4", count=V, offset=off)
    o = raw.astype(np.float32)                          # o head 0 [128]

    # z (head 0) for the same warm token (11751)
    hn = rmsnorm(W["embed_tokens.weight"][11751].astype(np.float32),
                 W["layers.0.input_layernorm.weight"], cfg["rms_norm_eps"])
    z = (W[f"{p}.in_proj_z.weight"] @ hn).astype(np.float32)[:V]
    w = W[f"{p}.norm.weight"].astype(np.float32)

    eps = cfg["rms_norm_eps"]
    var = np.mean(o * o)
    on = o * (1.0 / np.sqrt(var + eps))
    y = (w * on * silu(z)).astype(np.float32)

    xs = np.linspace(-16, 16, LUTN, endpoint=False)     # sigmoid LUT, left edges
    lut = sigmoid(xs).astype(np.float32)

    print(f"V={V}  o range [{o.min():.2e},{o.max():.2e}]  var={var:.2e}  z range [{z.min():.2f},{z.max():.2f}]")
    print(f"  y range [{y.min():.3f},{y.max():.3f}]")
    with open("artifacts/tv_gnorm.bin", "wb") as f:
        np.array([V, LUTN], "<i4").tofile(f)
        lut.tofile(f)
        o.tofile(f); w.tofile(f); z.tofile(f); y.tofile(f)
    print("wrote artifacts/tv_gnorm.bin")


if __name__ == "__main__":
    main()
