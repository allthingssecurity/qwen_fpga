"""Does int8 survive? Measure fp32 vs int8 decode, agreement + KL + first divergence.

The whole ~518 tok/s claim rests on the weights being 752 MB of int8. If int8
degrades the model, the premise is dead and we should be arguing about int4
groupwise or bf16 instead. So measure it before writing a line of RTL.

    python3 golden/eval_quant.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import DecodeState, decode_step  # noqa: E402
from quant import quantize_weights, rel_err  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "In 1969, humans first landed on",
    "The mitochondria is the",
]
N_NEW = 48


def logsoftmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def run(W, cfg, ids, n_new, force=None):
    """Greedy decode. If `force` is given, teacher-force those tokens and just
    record the logits -- so both models see an identical token stream and the
    comparison isolates numerics from divergence-amplification."""
    st = DecodeState.new(cfg)
    lg = None
    for t in ids:
        lg = decode_step(W, cfg, int(t), st)
    outs, logits = [], []
    for i in range(n_new):
        logits.append(lg)
        nxt = int(force[i]) if force is not None else int(np.argmax(lg))
        outs.append(int(np.argmax(lg)))
        lg = decode_step(W, cfg, nxt, st)
    return outs, np.stack(logits)


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    print("loading fp32 weights...")
    W = load_text_weights(snap)

    variants = {}
    print("\n[int8 all, incl. tied embed/lm_head]")
    variants["int8-all"] = quantize_weights(W, exclude_embed=False)[0]
    print("\n[int8 body only, embed/lm_head left fp32]")
    variants["int8-body"] = quantize_weights(W, exclude_embed=True)[0]

    print("\nworst per-tensor relative Frobenius error (int8-all), top 6:")
    for d, k in rel_err(W, variants["int8-all"])[:6]:
        print(f"  {d:.4%}  {k}")

    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(snap)
    except Exception:
        tk = None

    print(f"\n{'variant':<11} {'prompt':<32} {'top1':>6} {'meanKL':>9} {'maxKL':>9} {'1st div':>8}")
    print("-" * 82)

    agg = {k: [] for k in variants}
    for p in PROMPTS:
        ids = tk(p, return_tensors="np").input_ids[0].tolist() if tk else [760, 6511, 314]
        ref_tok, ref_lg = run(W, cfg, ids, N_NEW)
        if tk:
            print(f"{'fp32':<11} {p[:30]:<32} {'--':>6} {'--':>9} {'--':>9} {'--':>8}"
                  f"   -> {tk.decode(ref_tok)[:44]!r}")
        for name, Wq in variants.items():
            # teacher-force the fp32 token stream so KL is measured on the same states
            q_tok, q_lg = run(Wq, cfg, ids, N_NEW, force=ref_tok)
            agree = np.mean([a == b for a, b in zip(q_tok, ref_tok)])
            kls = []
            for a, b in zip(ref_lg, q_lg):
                la, lb = logsoftmax(a), logsoftmax(b)
                kls.append(float((np.exp(la) * (la - lb)).sum()))
            first = next((i for i, (a, b) in enumerate(zip(q_tok, ref_tok)) if a != b), -1)
            agg[name].append((agree, np.mean(kls), np.max(kls)))
            print(f"{name:<11} {p[:30]:<32} {agree:>5.1%} {np.mean(kls):>9.2e} {np.max(kls):>9.2e} "
                  f"{(first if first >= 0 else 'none'):>8}")

    print("-" * 82)
    for name, rows in agg.items():
        a = np.mean([r[0] for r in rows])
        mk = np.mean([r[1] for r in rows])
        xk = np.max([r[2] for r in rows])
        print(f"{name:<11} mean top-1 agreement {a:6.1%}   mean KL {mk:.3e}   worst KL {xk:.3e}")


if __name__ == "__main__":
    main()
