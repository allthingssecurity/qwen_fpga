"""One consistent 16-head dataset for the full DeltaNet recurrence stage of layer 0,
token 11751 (warm state from tv_deltanet). Everything the mixer core needs for all
heads, plus the golden per-head gated-norm output.

  python3 scripts/export_mixer16_tv.py  ->  artifacts/tv_mixer16.bin
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import rmsnorm, sigmoid, silu, softplus  # noqa: E402

H, K, V, LUTN = 16, 128, 128, 1024


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)
    p = "layers.0.linear_attn"
    eps = cfg["rms_norm_eps"]

    # q,k,v (post-l2norm, all heads) and warm state from tv_deltanet
    meta = {t["name"]: t for t in json.load(open("artifacts/tv_deltanet.json"))["tensors"]}
    def rd(name, n):
        return np.fromfile("artifacts/tv_deltanet.bin", "<f4", count=n, offset=meta[name]["offset"])
    q = rd("q", H*K).reshape(H, K); k = rd("k", H*K).reshape(H, K); v = rd("v", H*V).reshape(H, V)
    S = rd("S_in", H*K*V).reshape(H, K, V)

    # gate inputs (all heads) for token 11751
    hn = rmsnorm(W["embed_tokens.weight"][11751].astype(np.float32),
                 W["layers.0.input_layernorm.weight"], eps)
    a = (W[f"{p}.in_proj_a.weight"] @ hn).astype(np.float32)
    b = (W[f"{p}.in_proj_b.weight"] @ hn).astype(np.float32)
    A = np.exp(W[f"{p}.A_log"].astype(np.float32))
    dt = W[f"{p}.dt_bias"].astype(np.float32)
    z = (W[f"{p}.in_proj_z.weight"] @ hn).astype(np.float32).reshape(H, V)
    w = W[f"{p}.norm.weight"].astype(np.float32)

    # golden recurrence + gated norm per head -> og[H,V]
    beta = sigmoid(b); g = -A * softplus(a + dt)
    og = np.zeros((H, V), np.float32)
    for h in range(H):
        Sh = S[h].copy()
        Sh *= np.exp(g[h])
        kv = np.einsum("kv,k->v", Sh, k[h])
        delta = (v[h] - kv) * beta[h]
        Sh += np.einsum("k,v->kv", k[h], delta)
        o = np.einsum("kv,k->v", Sh, q[h])
        var = np.mean(o * o)
        on = o * (1.0 / np.sqrt(var + eps))
        og[h] = w * on * silu(z[h])

    xs = np.linspace(-16, 16, LUTN, endpoint=False)
    lut_sp = softplus(xs).astype(np.float32)
    lut_sg = sigmoid(xs).astype(np.float32)
    lut_ex = np.exp(np.linspace(-16, 0, LUTN, endpoint=False)).astype(np.float32)

    print(f"H={H}  og range [{og.min():.3f},{og.max():.3f}]")
    with open("artifacts/tv_mixer16.bin", "wb") as f:
        np.array([H, K, LUTN], "<i4").tofile(f)
        lut_sp.tofile(f); lut_ex.tofile(f); lut_sg.tofile(f)
        a.astype("<f4").tofile(f); b.astype("<f4").tofile(f)
        A.astype("<f4").tofile(f); dt.astype("<f4").tofile(f)
        q.astype("<f4").tofile(f); k.astype("<f4").tofile(f); v.astype("<f4").tofile(f)
        S.astype("<f4").tofile(f); z.astype("<f4").tofile(f)
        w.astype("<f4").tofile(f); og.astype("<f4").tofile(f)
    print("wrote artifacts/tv_mixer16.bin")


if __name__ == "__main__":
    main()
