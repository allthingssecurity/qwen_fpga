"""Full attention mixer dataset, self-consistent. Run golden full_attention for
layer 3 standalone over warm-up tokens to build a KV cache, then token 11751,
capturing all weights/inputs the RTL chain needs, the warm cache, and the golden
output.

Datapath: hn -> q/k/v_proj (int8) -> QK-norm -> RoPE -> [append to cache] ->
scores -> softmax -> context -> output gate -> o_proj.

  python3 scripts/export_attnfull_tv.py  ->  artifacts/tv_attnfull.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
import model as M  # noqa: E402
from model import DecodeState, full_attention, rmsnorm, rope_cos_sin  # noqa: E402
from quant import quant_rows  # noqa: E402

LUTN = 1024


def q_i8_vec(x):
    amax = np.abs(x).max(); s = (amax / 127.0) if amax > 0 else 1.0
    return np.rint(x / s).clip(-127, 127).astype(np.int8), np.float32(s)


def main():
    snap = find_snapshot(); cfg = load_config(snap); W = load_text_weights(snap, verbose=False)
    p = "layers.3.self_attn"; iln = W["layers.3.input_layernorm.weight"]; eps = cfg["rms_norm_eps"]
    HD, NH, NKV = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    M.ACT_INT8 = False
    st = DecodeState.new(cfg)

    def attn_in(tok):
        return rmsnorm(W["embed_tokens.weight"][tok].astype(np.float32), iln, eps)

    warm = [760, 6511, 314, 9338, 369]
    for pos, tok in enumerate(warm):
        full_attention(W, p, cfg, attn_in(tok), st, 0, pos)

    kc_warm = st.kc[0].copy()   # [T,2,256]
    vc_warm = st.vc[0].copy()
    T = kc_warm.shape[0]
    hn = attn_in(11751)
    out_golden = full_attention(W, p, cfg, hn, st, 0, len(warm)).astype(np.float32)  # [1024]

    x_i8, sx = q_i8_vec(hn)
    wq, swq = quant_rows(W[f"{p}.q_proj.weight"])   # [4096,1024]
    wk, swk = quant_rows(W[f"{p}.k_proj.weight"])   # [512,1024]
    wv, swv = quant_rows(W[f"{p}.v_proj.weight"])   # [512,1024]
    wo, swo = quant_rows(W[f"{p}.o_proj.weight"])   # [1024,2048]
    qn = W[f"{p}.q_norm.weight"].astype(np.float32); kn = W[f"{p}.k_norm.weight"].astype(np.float32)
    rp = cfg["rope_parameters"]
    cos, sin = rope_cos_sin(len(warm), HD, rp.get("partial_rotary_factor", 0.25), rp["rope_theta"])
    xs = np.linspace(-16, 16, LUTN, endpoint=False)
    lut_sg = 1.0 / (1.0 + np.exp(-xs))
    lut_ex = np.exp(np.linspace(-16, 0, LUTN, endpoint=False))

    print(f"T(warm)={T}  HD={HD} NH={NH} NKV={NKV}  out range [{out_golden.min():.3f},{out_golden.max():.3f}]")
    with open("artifacts/tv_attnfull.bin", "wb") as f:
        np.array([T, HD, NH, NKV, LUTN], "<i4").tofile(f)
        lut_ex.astype("<f4").tofile(f); lut_sg.astype("<f4").tofile(f)
        cos.astype("<f4").tofile(f); sin.astype("<f4").tofile(f)
        x_i8.astype("<i1").tofile(f); np.array([sx], "<f4").tofile(f)
        for wi, sw in [(wq, swq), (wk, swk), (wv, swv), (wo, swo)]:
            wi.reshape(-1).astype("<i1").tofile(f); sw.astype("<f4").tofile(f)
        qn.tofile(f); kn.tofile(f)
        kc_warm.astype("<f4").tofile(f); vc_warm.astype("<f4").tofile(f)
        out_golden.tofile(f)
    print("wrote artifacts/tv_attnfull.bin")


if __name__ == "__main__":
    main()
