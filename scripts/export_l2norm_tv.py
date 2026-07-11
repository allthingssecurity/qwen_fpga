"""Export a real L2-norm case (as used on q,k in DeltaNet): y = x*rsqrt(sum(x^2)+eps)*scale.
For q the scale is 1/sqrt(128); L2-norm is input-generic so a real vector exercises it.

  python3 scripts/export_l2norm_tv.py  ->  artifacts/tv_l2norm.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import l2norm, rmsnorm  # noqa: E402

K = 128


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)
    p = "layers.0.linear_attn"
    hn = rmsnorm(W["embed_tokens.weight"][6511].astype(np.float32),
                 W["layers.0.input_layernorm.weight"], cfg["rms_norm_eps"])
    qkv = (W[f"{p}.in_proj_qkv.weight"] @ hn).astype(np.float32)
    x = qkv[:K]                                    # a real q-head-sized vector
    scale = np.float32(1.0 / np.sqrt(K))
    y = (l2norm(x) * scale).astype(np.float32)

    print(f"K={K}  x range [{x.min():.2f},{x.max():.2f}]  sum(x^2)={np.sum(x*x):.2f}  "
          f"y range [{y.min():.4f},{y.max():.4f}]")
    with open("artifacts/tv_l2norm.bin", "wb") as f:
        x.astype("<f4").tofile(f)
        y.astype("<f4").tofile(f)
    print("wrote artifacts/tv_l2norm.bin")


if __name__ == "__main__":
    main()
