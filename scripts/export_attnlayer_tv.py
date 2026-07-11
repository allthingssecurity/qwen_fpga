"""Full attention decoder layer (layer 3), self-consistent, end to end:
   r = h;  h = r + full_attention(rmsnorm(h, input_ln))
   r = h;  h = r + mlp(rmsnorm(h, post_attention_ln))
Warms the KV cache over a few tokens, then processes token 11751 as a full layer.
Exports h_in, all weights (attention + MLP), the warm cache, and golden h_out.

  python3 scripts/export_attnlayer_tv.py  ->  artifacts/tv_attnlayer.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
import model as M  # noqa: E402
from model import DecodeState, full_attention, mlp, rmsnorm, rope_cos_sin  # noqa: E402
from quant import quant_rows  # noqa: E402

LUTN = 1024


def q_i8_vec(x):
    amax = np.abs(x).max(); s = (amax / 127.0) if amax > 0 else 1.0
    return np.rint(x / s).clip(-127, 127).astype(np.int8), np.float32(s)


def main():
    snap = find_snapshot(); cfg = load_config(snap); W = load_text_weights(snap, verbose=False)
    p = "layers.3.self_attn"; iln = W["layers.3.input_layernorm.weight"]
    pln = W["layers.3.post_attention_layernorm.weight"]; eps = cfg["rms_norm_eps"]
    HD, NH, NKV = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    M.ACT_INT8 = False
    st = DecodeState.new(cfg)

    def attn_in(tok):
        return rmsnorm(W["embed_tokens.weight"][tok].astype(np.float32), iln, eps)

    warm = [760, 6511, 314, 9338, 369]
    for pos, tok in enumerate(warm):
        full_attention(W, p, cfg, attn_in(tok), st, 0, pos)
    kc_warm = st.kc[0].copy(); vc_warm = st.vc[0].copy(); T = kc_warm.shape[0]

    h_in = W["embed_tokens.weight"][11751].astype(np.float32)
    hn = attn_in(11751)
    mix = full_attention(W, p, cfg, hn, st, 0, len(warm))
    h1 = h_in + mix
    hn2 = rmsnorm(h1, pln, eps)
    mlp_out = mlp(W, "layers.3.mlp", hn2)
    h_out = (h1 + mlp_out).astype(np.float32)

    x_i8, sx = q_i8_vec(hn)
    wq, swq = quant_rows(W[f"{p}.q_proj.weight"]); wk, swk = quant_rows(W[f"{p}.k_proj.weight"])
    wv, swv = quant_rows(W[f"{p}.v_proj.weight"]); wo, swo = quant_rows(W[f"{p}.o_proj.weight"])
    qn = W[f"{p}.q_norm.weight"].astype(np.float32); kn = W[f"{p}.k_norm.weight"].astype(np.float32)
    rp = cfg["rope_parameters"]
    cos, sin = rope_cos_sin(len(warm), HD, rp.get("partial_rotary_factor", 0.25), rp["rope_theta"])
    wg, swg = quant_rows(W["layers.3.mlp.gate_proj.weight"])
    wu, swu = quant_rows(W["layers.3.mlp.up_proj.weight"])
    wd, swd = quant_rows(W["layers.3.mlp.down_proj.weight"])
    lut_sg = 1.0 / (1.0 + np.exp(-np.linspace(-16, 16, LUTN, endpoint=False)))
    lut_ex = np.exp(np.linspace(-16, 0, LUTN, endpoint=False))

    print(f"T(warm)={T}  out range [{h_out.min():.3f},{h_out.max():.3f}]")
    with open("artifacts/tv_attnlayer.bin", "wb") as f:
        np.array([T, HD, NH, NKV, LUTN], "<i4").tofile(f)
        lut_ex.astype("<f4").tofile(f); lut_sg.astype("<f4").tofile(f)
        cos.astype("<f4").tofile(f); sin.astype("<f4").tofile(f)
        h_in.tofile(f)
        x_i8.astype("<i1").tofile(f); np.array([sx], "<f4").tofile(f)
        for wi, sw in [(wq, swq), (wk, swk), (wv, swv), (wo, swo)]:
            wi.reshape(-1).astype("<i1").tofile(f); sw.astype("<f4").tofile(f)
        qn.tofile(f); kn.tofile(f)
        kc_warm.astype("<f4").tofile(f); vc_warm.astype("<f4").tofile(f)
        pln.astype("<f4").tofile(f)
        for wi, sw in [(wg, swg), (wu, swu), (wd, swd)]:
            wi.reshape(-1).astype("<i1").tofile(f); sw.astype("<f4").tofile(f)
        mix.astype("<f4").tofile(f)                 # golden attention mixer output (pre-residual)
        hn2.astype("<f4").tofile(f)                 # golden post-norm input to MLP
        h_out.tofile(f)
    print("wrote artifacts/tv_attnlayer.bin")


if __name__ == "__main__":
    main()
