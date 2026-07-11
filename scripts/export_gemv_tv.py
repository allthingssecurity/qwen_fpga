"""Export a bit-exact int8 GEMV test case from a real Qwen layer, for the RTL
testbench. The core hardware op is integer: acc[o] = sum_i w_i8[o,i] * x_i8[i].
That is pure integer arithmetic, so the RTL must match numpy exactly, no
tolerance. Also exports the per-row scales and the dequantised fp32 result for
the later dequant check.

Uses a real full-attention k_proj weight (512 x 1024) and a real normed
activation, both int8 quantised the way the model does it.

  python3 scripts/export_gemv_tv.py   ->  artifacts/tv_gemv.{bin,json}
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import rmsnorm  # noqa: E402
from quant import quant_rows  # noqa: E402


def q_i8_vec(x):
    """per-vector symmetric int8, returns (q_i8, scale)."""
    amax = np.abs(x).max()
    s = (amax / 127.0) if amax > 0 else 1.0
    q = np.rint(x / s).clip(-127, 127).astype(np.int8)
    return q, np.float32(s)


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)

    # a real activation: embed a token, apply the layer-3 input norm (1+w)
    tok = 6511
    h = W["embed_tokens.weight"][tok].astype(np.float32)
    hn = rmsnorm(h, W["layers.3.input_layernorm.weight"], cfg["rms_norm_eps"])
    x_i8, sx = q_i8_vec(hn)                       # [1024] int8, scalar scale

    # real k_proj weights, int8 per output row (as packed)
    Wk = W["layers.3.self_attn.k_proj.weight"]    # [512, 1024] fp32
    w_i8, sw = quant_rows(Wk)                      # int8 [512,1024], scale [512]

    OUT, IN = w_i8.shape
    acc_i32 = (w_i8.astype(np.int32) @ x_i8.astype(np.int32)).astype(np.int32)  # exact
    y_fp32 = acc_i32.astype(np.float32) * sx * sw                                # dequant

    print(f"GEMV test case: OUT={OUT} IN={IN}")
    print(f"  x_i8 range [{x_i8.min()},{x_i8.max()}]  sx={sx:.6e}")
    print(f"  acc_i32 range [{acc_i32.min()},{acc_i32.max()}]")
    print(f"  y_fp32 range [{y_fp32.min():.4f},{y_fp32.max():.4f}]")

    os.makedirs("artifacts", exist_ok=True)
    # flat binary for the C++ testbench: x_i8[IN], w_i8[OUT*IN], acc_i32[OUT],
    # sx (f4), sw[OUT] (f4), y_fp32[OUT]
    with open("artifacts/tv_gemv.bin", "wb") as f:
        x_i8.astype("<i1").tofile(f)
        w_i8.reshape(-1).astype("<i1").tofile(f)
        acc_i32.astype("<i4").tofile(f)
        np.array([sx], "<f4").tofile(f)
        sw.astype("<f4").tofile(f)
        y_fp32.astype("<f4").tofile(f)
    with open("artifacts/tv_gemv.json", "w") as f:
        json.dump({"IN": IN, "OUT": OUT}, f)
    print("wrote artifacts/tv_gemv.bin + .json")


if __name__ == "__main__":
    main()
