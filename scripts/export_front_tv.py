"""Front of the DeltaNet mixer: hidden -> in_proj_qkv (int8 matvec) -> causal conv.
Exports the int8 activation and weights for the matvec, the conv weights and a warm
conv state, and the golden conv output (pre-silu) for token 11751, layer 0.

  python3 scripts/export_front_tv.py  ->  artifacts/tv_front.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import rmsnorm  # noqa: E402
from quant import quant_rows  # noqa: E402

CONV_DIM, Kk = 6144, 4


def q_i8_vec(x):
    amax = np.abs(x).max()
    s = (amax / 127.0) if amax > 0 else 1.0
    return np.rint(x / s).clip(-127, 127).astype(np.int8), np.float32(s)


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)
    p = "layers.0.linear_attn"
    iln = W["layers.0.input_layernorm.weight"]
    Wqkv = W[f"{p}.in_proj_qkv.weight"]            # [6144,1024] fp32
    cw = W[f"{p}.conv1d.weight"][:, 0, :]          # [6144,4]
    wi8, sw = quant_rows(Wqkv)                     # int8 [6144,1024], scale [6144]

    # warm the conv state over a few tokens, capture the state entering 11751
    state = np.zeros((CONV_DIM, Kk), np.float32)
    for t in [760, 6511, 314, 9338, 369]:
        x = rmsnorm(W["embed_tokens.weight"][t].astype(np.float32), iln, cfg["rms_norm_eps"])
        qkv = (Wqkv @ x).astype(np.float32)
        state = np.concatenate([state, qkv[:, None]], axis=1)[:, -4:]
    state_in = state.copy()

    # token 11751: int8 activation for the matvec, and golden qkv + conv_pre
    hn = rmsnorm(W["embed_tokens.weight"][11751].astype(np.float32), iln, cfg["rms_norm_eps"])
    x_i8, sx = q_i8_vec(hn)
    qkv = (Wqkv @ hn).astype(np.float32)                   # golden in_proj output
    xn = np.concatenate([state_in, qkv[:, None]], axis=1)  # [6144,5]
    conv_pre = np.sum(xn[:, 1:5] * cw, axis=1).astype(np.float32)

    print(f"CONV_DIM={CONV_DIM}  qkv range [{qkv.min():.2f},{qkv.max():.2f}]  "
          f"conv_pre range [{conv_pre.min():.2f},{conv_pre.max():.2f}]")
    with open("artifacts/tv_front.bin", "wb") as f:
        np.array([CONV_DIM, 1024], "<i4").tofile(f)
        x_i8.astype("<i1").tofile(f)
        np.array([sx], "<f4").tofile(f)
        wi8.reshape(-1).astype("<i1").tofile(f)
        sw.astype("<f4").tofile(f)
        cw.astype("<f4").tofile(f)          # [6144,4]
        state_in.astype("<f4").tofile(f)    # [6144,4]
        conv_pre.astype("<f4").tofile(f)    # [6144]
    print("wrote artifacts/tv_front.bin")


if __name__ == "__main__":
    main()
