"""What does the int8 scheme itself (per-row int8 weights, per-vector int8
activations, fp32 arithmetic) do over all 24 layers, independent of any RTL or
fixed-point rounding? This is the accuracy ceiling the RTL can reach. If this
predicts a different token than the fp32 model, no hardware can do better without
a better quantization scheme (per-layer calibration, higher precision, etc.).

  python3 scripts/int8_ceiling.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
import model as M  # noqa: E402
from model import DecodeState, rmsnorm, rope_cos_sin, silu, sigmoid, softplus, l2norm  # noqa: E402
from quant import quant_rows  # noqa: E402

WARM = [760, 6511, 314, 9338, 369]
TARGET = 11751


def qa(x):  # per-vector int8 activation round-trip
    a = np.abs(x).max(); s = (a / 127.0) if a > 0 else 1.0
    return np.rint(x / s).clip(-127, 127) * s


def qw(w):  # per-row int8 weight round-trip
    qi, s = quant_rows(w); return qi.astype(np.float32) * s[:, None]


def mv(W, key, x, quantw):
    w = qw(W[key]) if quantw else W[key]
    return w @ qa(x)


def gdn(W, p, cfg, x, st, li, qwt):
    H, K = cfg["linear_num_value_heads"], cfg["linear_key_head_dim"]
    kd = K * cfg["linear_num_key_heads"]; eps = cfg["rms_norm_eps"]
    qkv = mv(W, f"{p}.in_proj_qkv.weight", x, qwt)
    z = W[f"{p}.in_proj_z.weight"] @ qa(x); b = W[f"{p}.in_proj_b.weight"] @ qa(x); a = W[f"{p}.in_proj_a.weight"] @ qa(x)
    cw = W[f"{p}.conv1d.weight"][:, 0, :]
    xn = np.concatenate([st.conv[li], qkv[:, None]], axis=1); st.conv[li] = xn[:, -4:]
    qkv = silu(np.sum(xn[:, 1:5] * cw, axis=1))
    q = qkv[:kd].reshape(H, K); k = qkv[kd:2 * kd].reshape(H, K); v = qkv[2 * kd:].reshape(H, K)
    beta = sigmoid(b); g = -np.exp(W[f"{p}.A_log"].astype(np.float32)) * softplus(a + W[f"{p}.dt_bias"])
    q = l2norm(q) * (K ** -0.5); k = l2norm(k)
    S = st.rec[li]; S *= np.exp(g)[:, None, None]
    kv = np.einsum("hkv,hk->hv", S, k); delta = (v - kv) * beta[:, None]; S += np.einsum("hk,hv->hkv", k, delta)
    o = np.einsum("hkv,hk->hv", S, q)
    var = np.mean(o ** 2, axis=-1, keepdims=True); o = o * (1.0 / np.sqrt(var + eps))
    o = W[f"{p}.norm.weight"] * o; o = o * silu(z.reshape(H, K))
    return mv(W, f"{p}.out_proj.weight", o.reshape(-1), qwt)


def fa(W, p, cfg, x, st, fi, pos, qwt):
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    qg = mv(W, f"{p}.q_proj.weight", x, qwt).reshape(nh, hd * 2)
    q, gate = qg[:, :hd], qg[:, hd:].reshape(-1)
    q = rmsnorm(q, W[f"{p}.q_norm.weight"], eps)
    k = rmsnorm(mv(W, f"{p}.k_proj.weight", x, qwt).reshape(nkv, hd), W[f"{p}.k_norm.weight"], eps)
    v = mv(W, f"{p}.v_proj.weight", x, qwt).reshape(nkv, hd)
    rp = cfg["rope_parameters"]; cos, sin = rope_cos_sin(pos, hd, rp.get("partial_rotary_factor", 0.25), rp["rope_theta"])
    q = M.apply_rope(q, cos, sin); k = M.apply_rope(k, cos, sin)
    st.kc[fi] = np.concatenate([st.kc[fi], k[None]], axis=0); st.vc[fi] = np.concatenate([st.vc[fi], v[None]], axis=0)
    rep = nh // nkv; Kc = np.repeat(st.kc[fi], rep, axis=1); Vc = np.repeat(st.vc[fi], rep, axis=1)
    s = np.einsum("hd,thd->ht", q, Kc) * (hd ** -0.5); s = s - s.max(-1, keepdims=True)
    w = np.exp(s); w /= w.sum(-1, keepdims=True)
    o = np.einsum("ht,thd->hd", w, Vc).reshape(-1) * sigmoid(gate)
    return mv(W, f"{p}.o_proj.weight", o, qwt)


def run(W, cfg, tok, st, qwt):
    eps = cfg["rms_norm_eps"]; h = W["embed_tokens.weight"][tok].copy().astype(np.float32); li = fi = 0
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"layers.{i}"; r = h; h = rmsnorm(h, W[f"{p}.input_layernorm.weight"], eps)
        if kind == "linear_attention":
            h = gdn(W, f"{p}.linear_attn", cfg, h, st, li, qwt); li += 1
        else:
            h = fa(W, f"{p}.self_attn", cfg, h, st, fi, st.pos, qwt); fi += 1
        h = r + h; r = h
        h = rmsnorm(h, W[f"{p}.post_attention_layernorm.weight"], eps)
        g = silu(mv(W, f"{p}.mlp.gate_proj.weight", h, qwt)) * mv(W, f"{p}.mlp.up_proj.weight", h, qwt)
        h = r + mv(W, f"{p}.mlp.down_proj.weight", g, qwt)
    st.pos += 1
    hf = rmsnorm(h, W["norm.weight"], eps)
    return W["embed_tokens.weight"] @ hf


def main():
    import copy
    snap = find_snapshot(); cfg = load_config(snap); W = load_text_weights(snap, verbose=False)
    M.ACT_INT8 = False
    # fp32 reference argmax
    st = DecodeState.new(cfg)
    for t in WARM:
        M.decode_step(W, cfg, t, st)
    fp32 = M.decode_step(W, cfg, TARGET, copy.deepcopy(st))
    print(f"fp32 argmax {int(np.argmax(fp32))}")
    # int8 activations only (fp32 weights)
    st2 = DecodeState.new(cfg)
    for t in WARM:
        run(W, cfg, t, st2, qwt=False)
    a8 = run(W, cfg, TARGET, copy.deepcopy(st2), qwt=False)
    print(f"int8 activations only  argmax {int(np.argmax(a8))}")
    # int8 weights + int8 activations (what the RTL does)
    st3 = DecodeState.new(cfg)
    for t in WARM:
        run(W, cfg, t, st3, qwt=True)
    w8 = run(W, cfg, TARGET, copy.deepcopy(st3), qwt=True)
    print(f"int8 weights + int8 acts  argmax {int(np.argmax(w8))}  (top5 {list(np.argsort(w8)[-5:][::-1])})")


if __name__ == "__main__":
    main()
