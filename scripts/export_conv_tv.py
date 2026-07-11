"""Export a warm causal-conv1d case from the DeltaNet front. Depthwise conv,
kernel 4: window = [state[1], state[2], state[3], new], out_pre = sum(window*cw).
Runs a few tokens to warm the state, captures the last step.

  python3 scripts/export_conv_tv.py  ->  artifacts/tv_conv.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import rmsnorm  # noqa: E402

CONV_DIM = 6144
K = 4


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)
    Wqkv = W["layers.0.linear_attn.in_proj_qkv.weight"]         # [6144,1024]
    cw = W["layers.0.linear_attn.conv1d.weight"][:, 0, :]        # [6144,4]
    iln = W["layers.0.input_layernorm.weight"]

    state = np.zeros((CONV_DIM, K), np.float32)
    new_qkv = state_in = conv_pre = None
    for t in [760, 6511, 314, 9338, 369]:
        x = rmsnorm(W["embed_tokens.weight"][t].astype(np.float32), iln, cfg["rms_norm_eps"])
        qkv = (Wqkv @ x).astype(np.float32)                     # [6144]
        state_in = state.copy()
        xn = np.concatenate([state, qkv[:, None]], axis=1)      # [6144,5]
        state = xn[:, -4:]
        conv_pre = np.sum(xn[:, 1:5] * cw, axis=1).astype(np.float32)
        new_qkv = qkv

    print(f"CONV_DIM={CONV_DIM}  qkv range [{new_qkv.min():.2f},{new_qkv.max():.2f}]  "
          f"cw range [{cw.min():.3f},{cw.max():.3f}]  conv_pre range [{conv_pre.min():.2f},{conv_pre.max():.2f}]")
    with open("artifacts/tv_conv.bin", "wb") as f:
        new_qkv.astype("<f4").tofile(f)
        state_in.astype("<f4").tofile(f)     # [6144,4]
        cw.astype("<f4").tofile(f)           # [6144,4]
        conv_pre.astype("<f4").tofile(f)     # [6144]
    print("wrote artifacts/tv_conv.bin")


if __name__ == "__main__":
    main()
