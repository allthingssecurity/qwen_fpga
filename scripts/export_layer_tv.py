"""Full DeltaNet decoder layer (layer 0), self-consistent, for end-to-end RTL check:
   r  = h
   h  = r + deltanet_mixer(rmsnorm(h, input_ln))
   r  = h
   h  = r + mlp(rmsnorm(h, post_attn_ln))            mlp = down(silu(gate(x))*up(x))
Warms the DeltaNet state over a few tokens, then processes token 11751 as a full
layer. Exports h_in, every weight, the warm state, and the golden layer output.

  python3 scripts/export_layer_tv.py  ->  artifacts/tv_layer.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
import model as M  # noqa: E402
from model import DecodeState, gated_deltanet, mlp, rmsnorm, sigmoid, softplus  # noqa: E402
from quant import quant_rows  # noqa: E402

H, K, CD, HID, INTER, LUTN = 16, 128, 6144, 1024, 3584, 1024


def q_i8_vec(x):
    amax = np.abs(x).max(); s = (amax / 127.0) if amax > 0 else 1.0
    return np.rint(x / s).clip(-127, 127).astype(np.int8), np.float32(s)


def main():
    snap = find_snapshot(); cfg = load_config(snap); W = load_text_weights(snap, verbose=False)
    la = "layers.0.linear_attn"; iln = W["layers.0.input_layernorm.weight"]
    pln = W["layers.0.post_attention_layernorm.weight"]; eps = cfg["rms_norm_eps"]
    M.ACT_INT8 = False
    st = DecodeState.new(cfg)

    def warm_mix(tok):
        gated_deltanet(W, la, cfg, rmsnorm(W["embed_tokens.weight"][tok].astype(np.float32), iln, eps), st, 0)

    for tok in [760, 6511, 314, 9338, 369]:
        warm_mix(tok)
    conv_state_in = st.conv[0].copy(); rec_state_in = st.rec[0].copy()

    # full layer for token 11751
    h_in = W["embed_tokens.weight"][11751].astype(np.float32)
    hn = rmsnorm(h_in, iln, eps)
    mix = gated_deltanet(W, la, cfg, hn, st, 0)
    h1 = h_in + mix
    hn2 = rmsnorm(h1, pln, eps)
    mlp_out = mlp(W, "layers.0.mlp", hn2)
    h_out = (h1 + mlp_out).astype(np.float32)

    # weights the RTL needs
    x_i8, sx = q_i8_vec(hn)
    wqkv, swq = quant_rows(W[f"{la}.in_proj_qkv.weight"])
    cw = W[f"{la}.conv1d.weight"][:, 0, :].astype(np.float32)
    a = (W[f"{la}.in_proj_a.weight"] @ hn).astype(np.float32)
    b = (W[f"{la}.in_proj_b.weight"] @ hn).astype(np.float32)
    A = np.exp(W[f"{la}.A_log"].astype(np.float32)); dt = W[f"{la}.dt_bias"].astype(np.float32)
    z = (W[f"{la}.in_proj_z.weight"] @ hn).astype(np.float32).reshape(H, K)
    nw = W[f"{la}.norm.weight"].astype(np.float32)
    wout, swo = quant_rows(W[f"{la}.out_proj.weight"])
    wg, swg = quant_rows(W["layers.0.mlp.gate_proj.weight"])   # [3584,1024]
    wu, swu = quant_rows(W["layers.0.mlp.up_proj.weight"])     # [3584,1024]
    wd, swd = quant_rows(W["layers.0.mlp.down_proj.weight"])   # [1024,3584]

    xs = np.linspace(-16, 16, LUTN, endpoint=False)
    lut_sp = softplus(xs).astype(np.float32); lut_sg = sigmoid(xs).astype(np.float32)
    lut_ex = np.exp(np.linspace(-16, 0, LUTN, endpoint=False)).astype(np.float32)

    print(f"h_out range [{h_out.min():.3f},{h_out.max():.3f}]")
    with open("artifacts/tv_layer.bin", "wb") as f:
        np.array([H, K, CD, HID, INTER, LUTN], "<i4").tofile(f)
        lut_sp.tofile(f); lut_ex.tofile(f); lut_sg.tofile(f)
        h_in.tofile(f)                                     # raw layer input (fp32, for residual)
        x_i8.astype("<i1").tofile(f); np.array([sx], "<f4").tofile(f)
        wqkv.reshape(-1).astype("<i1").tofile(f); swq.astype("<f4").tofile(f)
        cw.tofile(f); conv_state_in.astype("<f4").tofile(f)
        a.tofile(f); b.tofile(f); A.tofile(f); dt.tofile(f); z.astype("<f4").tofile(f); nw.tofile(f)
        rec_state_in.astype("<f4").tofile(f)
        wout.reshape(-1).astype("<i1").tofile(f); swo.astype("<f4").tofile(f)
        pln.astype("<f4").tofile(f)                        # post-attn layernorm weight
        for wi, sw in [(wg, swg), (wu, swu), (wd, swd)]:
            wi.reshape(-1).astype("<i1").tofile(f); sw.astype("<f4").tofile(f)
        h_out.tofile(f)
    print("wrote artifacts/tv_layer.bin")


if __name__ == "__main__":
    main()
