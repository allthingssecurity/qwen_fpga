"""Export a real SwiGLU elementwise case: gate and up projections of a real
hidden, and the golden silu(gate) * up. Also the sigmoid lookup table the RTL
uses. silu(x) = x * sigmoid(x).

  python3 scripts/export_swiglu_tv.py  ->  artifacts/tv_swiglu.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import rmsnorm, silu  # noqa: E402

LUTN = 1024
RANGE = 8.0   # sigmoid table covers [-8, 8]


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)

    x = rmsnorm(W["embed_tokens.weight"][6511].astype(np.float32),
                W["layers.3.post_attention_layernorm.weight"], cfg["rms_norm_eps"])
    g = W["layers.3.mlp.gate_proj.weight"] @ x      # [3584]
    u = W["layers.3.mlp.up_proj.weight"] @ x        # [3584]
    out = silu(g) * u

    # sigmoid LUT at bin centers over [-RANGE, RANGE], Q16
    idx = np.arange(LUTN)
    xs = -RANGE + (idx + 0.5) * (2 * RANGE / LUTN)
    lut = (1.0 / (1.0 + np.exp(-xs)))                # [0,1)

    print(f"INTER={g.size}  g range [{g.min():.2f},{g.max():.2f}]  u range [{u.min():.2f},{u.max():.2f}]")
    print(f"  out range [{out.min():.3f},{out.max():.3f}]")
    with open("artifacts/tv_swiglu.bin", "wb") as f:
        np.array([g.size, LUTN], "<i4").tofile(f)
        lut.astype("<f4").tofile(f)
        g.astype("<f4").tofile(f)
        u.astype("<f4").tofile(f)
        out.astype("<f4").tofile(f)
    print("wrote artifacts/tv_swiglu.bin")


if __name__ == "__main__":
    main()
