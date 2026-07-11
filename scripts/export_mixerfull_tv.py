"""Full DeltaNet mixer dataset, self-consistent: run golden gated_deltanet for
layer 0 standalone over warm-up tokens, then token 11751, capturing every input
and weight the RTL chain needs and the golden mixer output.

Main datapath the RTL computes: hn -> in_proj_qkv (int8 matvec) -> conv -> silu ->
split -> l2norm(q,k) -> [gate+recurrence+gated_norm per head] -> og -> out_proj.
The small side projections a,b,z (gate/gnorm inputs) are captured golden.

  python3 scripts/export_mixerfull_tv.py  ->  artifacts/tv_mixerfull.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
import model as M  # noqa: E402
from model import DecodeState, gated_deltanet, rmsnorm, sigmoid, softplus  # noqa: E402
from quant import quant_rows  # noqa: E402

H, K, CD, HID, LUTN = 16, 128, 6144, 1024, 1024


def q_i8_vec(x):
    amax = np.abs(x).max(); s = (amax / 127.0) if amax > 0 else 1.0
    return np.rint(x / s).clip(-127, 127).astype(np.int8), np.float32(s)


def main():
    snap = find_snapshot(); cfg = load_config(snap); W = load_text_weights(snap, verbose=False)
    p = "layers.0.linear_attn"; iln = W["layers.0.input_layernorm.weight"]; eps = cfg["rms_norm_eps"]
    M.ACT_INT8 = False
    st = DecodeState.new(cfg)

    def dn_input(tok):
        return rmsnorm(W["embed_tokens.weight"][tok].astype(np.float32), iln, eps)

    for tok in [760, 6511, 314, 9338, 369]:          # warm conv + recurrent state
        gated_deltanet(W, p, cfg, dn_input(tok), st, 0)

    conv_state_in = st.conv[0].copy()                 # [6144,4]
    rec_state_in = st.rec[0].copy()                   # [16,128,128]
    hn = dn_input(11751)
    out_golden = gated_deltanet(W, p, cfg, hn, st, 0).astype(np.float32)   # [1024]

    # inputs / weights the RTL needs
    x_i8, sx = q_i8_vec(hn)
    wqkv_i8, sw_qkv = quant_rows(W[f"{p}.in_proj_qkv.weight"])
    cw = W[f"{p}.conv1d.weight"][:, 0, :].astype(np.float32)
    a = (W[f"{p}.in_proj_a.weight"] @ hn).astype(np.float32)
    b = (W[f"{p}.in_proj_b.weight"] @ hn).astype(np.float32)
    A = np.exp(W[f"{p}.A_log"].astype(np.float32)); dt = W[f"{p}.dt_bias"].astype(np.float32)
    z = (W[f"{p}.in_proj_z.weight"] @ hn).astype(np.float32).reshape(H, K)
    nw = W[f"{p}.norm.weight"].astype(np.float32)
    wout_i8, sw_out = quant_rows(W[f"{p}.out_proj.weight"])   # [1024,2048]

    xs = np.linspace(-16, 16, LUTN, endpoint=False)
    lut_sp = softplus(xs).astype(np.float32); lut_sg = sigmoid(xs).astype(np.float32)
    lut_ex = np.exp(np.linspace(-16, 0, LUTN, endpoint=False)).astype(np.float32)

    print(f"out_golden range [{out_golden.min():.3f},{out_golden.max():.3f}]")
    with open("artifacts/tv_mixerfull.bin", "wb") as f:
        np.array([H, K, CD, HID, LUTN], "<i4").tofile(f)
        lut_sp.tofile(f); lut_ex.tofile(f); lut_sg.tofile(f)
        x_i8.astype("<i1").tofile(f); np.array([sx], "<f4").tofile(f)
        wqkv_i8.reshape(-1).astype("<i1").tofile(f); sw_qkv.astype("<f4").tofile(f)
        cw.tofile(f); conv_state_in.astype("<f4").tofile(f)
        a.tofile(f); b.tofile(f); A.tofile(f); dt.tofile(f)
        z.astype("<f4").tofile(f); nw.tofile(f)
        rec_state_in.astype("<f4").tofile(f)
        wout_i8.reshape(-1).astype("<i1").tofile(f); sw_out.astype("<f4").tofile(f)
        out_golden.tofile(f)
    print("wrote artifacts/tv_mixerfull.bin")


if __name__ == "__main__":
    main()
