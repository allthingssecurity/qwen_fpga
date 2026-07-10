"""Check the numpy golden model against the torch reference, step by step.

Prints per-step, per-layer max abs deviation so a divergence is localised to
the exact (token, layer) where it starts -- which is how you find a sign error
in a delta rule instead of staring at wrong logits.

    ./.venv/bin/python golden/dump_torch_ref.py     # produces artifacts/ref.npz
    python3 golden/verify.py                        # no torch needed here
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import DecodeState, decode_step  # noqa: E402

TOL_HIDDEN = 2e-3     # fp32 accumulation-order noise over 24 layers
TOL_LOGIT = 5e-3


def main():
    ref = np.load("artifacts/ref.npz")
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap)

    fed = ref["fed_tokens"]
    st = DecodeState.new(cfg)

    print(f"\n{'step':>4} {'tok':>7} {'max|dh| per layer':>20} {'max|dlogit|':>12} {'argmax':>16}")
    print("-" * 68)

    worst_h = worst_l = 0.0
    bad = []

    for s, t in enumerate(fed):
        # instrument decode_step by re-running its body with capture
        h_np, layer_devs = _decode_capture(W, cfg, int(t), st, ref, s)
        logits_np = h_np
        logits_ref = ref[f"s{s}_logits"]

        dl = float(np.abs(logits_np - logits_ref).max())
        dh = max(layer_devs) if layer_devs else 0.0
        worst_h, worst_l = max(worst_h, dh), max(worst_l, dl)

        a_np, a_ref = int(np.argmax(logits_np)), int(np.argmax(logits_ref))
        ok = "ok" if a_np == a_ref else "MISMATCH"
        if a_np != a_ref or dl > TOL_LOGIT or dh > TOL_HIDDEN:
            bad.append(s)
        print(f"{s:>4} {int(t):>7} {dh:>20.3e} {dl:>12.3e} {a_np:>7}/{a_ref:<7} {ok}")

    print("-" * 68)
    print(f"worst hidden dev {worst_h:.3e} (tol {TOL_HIDDEN:.0e})")
    print(f"worst logit  dev {worst_l:.3e} (tol {TOL_LOGIT:.0e})")

    # DeltaNet recurrent state check
    lt = cfg["layer_types"]
    lin = [i for i, k in enumerate(lt) if k == "linear_attention"]
    print("\nDeltaNet state:")
    smax = 0.0
    for j in range(len(lin)):
        if f"rec{j}" not in ref:
            continue
        d = float(np.abs(st.rec[j].reshape(-1) - ref[f"rec{j}"]).max())
        c = float(np.abs(st.conv[j].reshape(-1) - ref[f"conv{j}"]).max()) if f"conv{j}" in ref else 0.0
        smax = max(smax, d, c)
        if j < 3 or d > 1e-3:
            print(f"  linear layer {lin[j]:>2}: max|dS| {d:.3e}   max|dconv| {c:.3e}")
    print(f"  worst over all 18 layers: {smax:.3e}")

    ok = not bad and worst_l <= TOL_LOGIT and smax < 1e-2
    print("\n" + ("PASS - numpy golden matches torch" if ok else f"FAIL - diverged at steps {bad}"))
    return 0 if ok else 1


def _decode_capture(W, cfg, tok, st, ref, s):
    """decode_step, but comparing each layer's output to the torch hook."""
    from model import full_attention, gated_deltanet, mlp, rmsnorm

    eps = cfg["rms_norm_eps"]
    h = W["embed_tokens.weight"][tok].copy()
    li = fi = 0
    devs = []

    for i, kind in enumerate(cfg["layer_types"]):
        p = f"layers.{i}"
        r = h
        h = rmsnorm(h, W[f"{p}.input_layernorm.weight"], eps)
        if kind == "linear_attention":
            h = gated_deltanet(W, f"{p}.linear_attn", cfg, h, st, li); li += 1
        else:
            h = full_attention(W, f"{p}.self_attn", cfg, h, st, fi, st.pos); fi += 1
        h = r + h
        r = h
        h = rmsnorm(h, W[f"{p}.post_attention_layernorm.weight"], eps)
        h = r + mlp(W, f"{p}.mlp", h)

        k = f"s{s}_layer{i}"
        if k in ref:
            devs.append(float(np.abs(h - ref[k]).max()))

    st.pos += 1
    h = rmsnorm(h, W["norm.weight"], eps)
    return W["embed_tokens.weight"] @ h, devs


if __name__ == "__main__":
    sys.exit(main())
