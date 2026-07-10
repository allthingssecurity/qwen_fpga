"""Does int8 ACTIVATION (on top of int8 weights) hold up?

This is the open item from docs/synthesis_estimate.md: int8 activations cut the
DSP budget ~10x (int8xint8 = 2 MAC/DSP vs fp32 ~5 DSP/MAC), but add error. Same
top-1-agreement / KL protocol as eval_quant.py. If W8A8 tracks fp32, the ~850-DSP
(comfortable) design is justified; if it collapses, we stay fp32 (~8,500 DSP) or
revisit.

  python3 scripts/eval_act_int8.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
import model as M  # noqa: E402
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import DecodeState, decode_step  # noqa: E402
from quant import quantize_weights  # noqa: E402

PROMPTS = ["The capital of France is", "def quicksort(arr):",
           "In 1969, humans first landed on", "The mitochondria is the"]
N_NEW = 48


def logsoftmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def run(W, cfg, ids, n_new, force=None):
    """force=None -> greedy self-consistent stream (the reference). Otherwise
    teacher-force `force` so every variant sees identical states."""
    st = DecodeState.new(cfg)
    lg = None
    for t in ids:
        lg = decode_step(W, cfg, int(t), st)
    outs, logits = [], []
    for i in range(n_new):
        logits.append(lg)
        outs.append(int(np.argmax(lg)))
        nxt = int(force[i]) if force is not None else int(np.argmax(lg))
        lg = decode_step(W, cfg, nxt, st)
    return outs, np.stack(logits)


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap)
    Wq = quantize_weights(W, exclude_embed=False, verbose=False)[0]

    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(snap)
    except Exception:
        tk = None

    variants = [("W16A16 (fp32 ref)", W, False),
                ("W8A16 (int8 wt)", Wq, False),
                ("W8A8  (int8 wt+act)", Wq, True)]

    print(f"\n{'variant':<20} {'top-1':>7} {'meanKL':>9} {'maxKL':>9}")
    print("-" * 50)
    agg = {name: [] for name, _, _ in variants}
    for p in PROMPTS:
        ids = tk(p, return_tensors="np").input_ids[0].tolist() if tk else [760, 6511, 314]
        M.ACT_INT8 = False
        ref_tok, ref_lg = run(W, cfg, ids, N_NEW)              # greedy fp32 reference stream
        for name, Wv, act in variants:
            M.ACT_INT8 = act
            q_tok, q_lg = run(Wv, cfg, ids, N_NEW, ref_tok)
            agree = np.mean([a == b for a, b in zip(q_tok, ref_tok)])
            kl = [float((np.exp(logsoftmax(a)) * (logsoftmax(a) - logsoftmax(b))).sum())
                  for a, b in zip(ref_lg, q_lg)]
            agg[name].append((agree, np.mean(kl), np.max(kl)))
        M.ACT_INT8 = False

    for name in agg:
        a = np.mean([r[0] for r in agg[name]])
        mk = np.mean([r[1] for r in agg[name]])
        xk = np.max([r[2] for r in agg[name]])
        print(f"{name:<20} {a:>6.1%} {mk:>9.2e} {xk:>9.2e}")
    print("\n(top-1 = agreement with the fp32 token stream; W16A16 row is the self-check)")


if __name__ == "__main__":
    main()
