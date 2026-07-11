"""Export a real RMSNorm case for the RTL testbench: a real hidden vector, the
input-layernorm weight, and the golden normed output (the (1+w) convention).

  python3 scripts/export_rmsnorm_tv.py  ->  artifacts/tv_rmsnorm.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import rmsnorm  # noqa: E402


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)

    x = W["embed_tokens.weight"][6511].astype(np.float32)      # [1024]
    w = W["layers.3.input_layernorm.weight"].astype(np.float32)
    y = rmsnorm(x, w, cfg["rms_norm_eps"])                     # golden, (1+w)

    print(f"HIDDEN={x.size}  mean(x^2)={np.mean(x*x):.4e}  y range [{y.min():.3f},{y.max():.3f}]")
    with open("artifacts/tv_rmsnorm.bin", "wb") as f:
        x.astype("<f4").tofile(f)
        w.astype("<f4").tofile(f)
        y.astype("<f4").tofile(f)
    print("wrote artifacts/tv_rmsnorm.bin")


if __name__ == "__main__":
    main()
