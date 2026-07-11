"""Whole-model decode step, for the full 24-layer RTL forward. Warms every layer's
state with a short prompt via golden decode, snapshots the state, then dumps
everything the RTL controller co-sim needs to run one token through all 24 layers
([DeltaNet x3, attention] x6), the final norm, and the tied output head.

Unlike the single-layer exports, the gate-feed projections z, a, b cannot be
precomputed: they depend on each layer's hidden state, which the RTL produces as
it goes. So their weights are exported and the co-sim recomputes them per layer.

Per-layer records carry a magic marker so a misaligned read trips an assert.

  python3 scripts/export_model_tv.py  ->  artifacts/tv_model.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
import model as M  # noqa: E402
from model import DecodeState, rmsnorm, rope_cos_sin  # noqa: E402
from quant import quant_rows  # noqa: E402

LUTN = 1024
WARM = [760, 6511, 314, 9338, 369]
TARGET = 11751


def qv(x):
    amax = np.abs(x).max(); s = (amax / 127.0) if amax > 0 else 1.0
    return np.rint(x / s).clip(-127, 127).astype(np.int8), np.float32(s)


def wi8(f, w):
    qi, s = quant_rows(w); qi.reshape(-1).astype("<i1").tofile(f); s.astype("<f4").tofile(f)


def forward_capture(W, cfg, tok, st):
    """decode_step, but also return the hidden entering each layer and the final."""
    eps = cfg["rms_norm_eps"]; h = W["embed_tokens.weight"][tok].copy().astype(np.float32)
    trace = []; li = fi = 0
    for i, kind in enumerate(cfg["layer_types"]):
        trace.append(h.copy()); p = f"layers.{i}"; r = h
        h = rmsnorm(h, W[f"{p}.input_layernorm.weight"], eps)
        if kind == "linear_attention":
            h = M.gated_deltanet(W, f"{p}.linear_attn", cfg, h, st, li); li += 1
        else:
            h = M.full_attention(W, f"{p}.self_attn", cfg, h, st, fi, st.pos); fi += 1
        h = r + h; r = h
        h = rmsnorm(h, W[f"{p}.post_attention_layernorm.weight"], eps)
        h = r + M.mlp(W, f"{p}.mlp", h)
    st.pos += 1
    hf = rmsnorm(h, W["norm.weight"], eps); trace.append(hf.copy())
    logits = W["embed_tokens.weight"] @ hf
    return np.array(trace, np.float32), logits.astype(np.float32)


def main():
    snap = find_snapshot(); cfg = load_config(snap); W = load_text_weights(snap, verbose=False)
    M.ACT_INT8 = False
    types = cfg["layer_types"]; N = len(types); HID = cfg["hidden_size"]; V = W["embed_tokens.weight"].shape[0]
    HD = cfg["head_dim"]; T = len(WARM)
    st = DecodeState.new(cfg)
    for tok in WARM:
        M.decode_step(W, cfg, tok, st)
    st_snap = st.clone() if hasattr(st, "clone") else _clone(st)     # state before target
    trace, logits = forward_capture(W, cfg, TARGET, _clone(st_snap))
    argmax = int(np.argmax(logits))
    print(f"warm={T} target={TARGET} -> argmax {argmax}  logit {logits[argmax]:.3f}  hidden|max {np.abs(trace[-1]).max():.3f}")

    xs = np.linspace(-16, 16, LUTN, endpoint=False)
    lut_sp = M.softplus(xs).astype(np.float32); lut_sg = M.sigmoid(xs).astype(np.float32)
    lut_ex = np.exp(np.linspace(-16, 0, LUTN, endpoint=False)).astype(np.float32)
    rp = cfg["rope_parameters"]
    cos, sin = rope_cos_sin(T, HD, rp.get("partial_rotary_factor", 0.25), rp["rope_theta"])

    ei8, es = quant_rows(W["embed_tokens.weight"].astype(np.float32))   # tied head [V,1024]
    with open("artifacts/tv_model.bin", "wb") as f:
        np.array([0x51574E00, N, HID, V, LUTN, T, TARGET, argmax], "<i4").tofile(f)
        lut_sp.tofile(f); lut_ex.tofile(f); lut_sg.tofile(f)
        W["norm.weight"].astype("<f4").tofile(f)
        W["embed_tokens.weight"][TARGET].astype("<f4").tofile(f)      # input embedding row
        cos.astype("<f4").tofile(f); sin.astype("<f4").tofile(f)
        ei8.reshape(-1).astype("<i1").tofile(f); es.astype("<f4").tofile(f)
        logits.astype("<f4").tofile(f)
        trace.astype("<f4").tofile(f)                                  # [N+1, HID]
        li = fi = 0
        for i, kind in enumerate(types):
            p = f"layers.{i}"; dn = (kind == "linear_attention")
            np.array([0x4C000000 | i, 0 if dn else 1], "<i4").tofile(f)
            W[f"{p}.input_layernorm.weight"].astype("<f4").tofile(f)
            W[f"{p}.post_attention_layernorm.weight"].astype("<f4").tofile(f)
            if dn:
                q = f"{p}.linear_attn"
                wi8(f, W[f"{q}.in_proj_qkv.weight"])
                wi8(f, W[f"{q}.in_proj_z.weight"])
                W[f"{q}.in_proj_a.weight"].astype("<f4").tofile(f)
                W[f"{q}.in_proj_b.weight"].astype("<f4").tofile(f)
                W[f"{q}.conv1d.weight"][:, 0, :].astype("<f4").tofile(f)
                st_snap.conv[li].astype("<f4").tofile(f)
                np.exp(W[f"{q}.A_log"].astype(np.float32)).astype("<f4").tofile(f)
                W[f"{q}.dt_bias"].astype("<f4").tofile(f)
                W[f"{q}.norm.weight"].astype("<f4").tofile(f)
                st_snap.rec[li].astype("<f4").tofile(f)
                wi8(f, W[f"{q}.out_proj.weight"]); li += 1
            else:
                q = f"{p}.self_attn"
                for nm in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    wi8(f, W[f"{q}.{nm}.weight"])
                W[f"{q}.q_norm.weight"].astype("<f4").tofile(f)
                W[f"{q}.k_norm.weight"].astype("<f4").tofile(f)
                st_snap.kc[fi].astype("<f4").tofile(f); st_snap.vc[fi].astype("<f4").tofile(f); fi += 1
            for nm in ["gate_proj", "up_proj", "down_proj"]:
                wi8(f, W[f"{p}.mlp.{nm}.weight"])
            np.array([0x4C000000 | i], "<i4").tofile(f)               # end marker
    sz = os.path.getsize("artifacts/tv_model.bin")
    print(f"wrote artifacts/tv_model.bin  ({sz/1e6:.0f} MB)")


def _clone(st):
    import copy
    return copy.deepcopy(st)


if __name__ == "__main__":
    main()
