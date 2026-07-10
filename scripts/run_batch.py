"""Run a batch of prompts through the verified model path and record everything:
full prompt, full answer, token counts, prefill and decode timings, tokens/sec.
Writes artifacts/generations.md for the repo. Uses the numpy golden model in
W8A8 mode (the exact arithmetic the FPGA datapath uses).

    python3 scripts/run_batch.py
"""

import sys
import time

import numpy as np

sys.path.insert(0, "golden")
import model as M  # noqa: E402
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import DecodeState, decode_step  # noqa: E402
from quant import quantize_weights  # noqa: E402

PROMPTS = [
    ("What is the capital of France?", 32, False),
    ("Write a Python function to reverse a linked list.", 160, False),
    ("Explain in simple terms why the sky is blue.", 120, False),
    ("def fibonacci(n):", 80, True),
    ("List three benefits of regular exercise.", 120, False),
    ("Write a short story about a robot who learns to paint.", 300, False),
    ("Explain how a transformer neural network works, step by step.", 320, False),
    ("What are the main differences between TCP and UDP?", 200, False),
]


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    print("loading weights...", flush=True)
    W = load_text_weights(snap, verbose=False)
    W = quantize_weights(W, exclude_embed=False, verbose=False)[0]
    M.ACT_INT8 = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    eos = {tok.eos_token_id, cfg.get("eos_token_id")}
    eos.discard(None)

    out = ["# Generations\n",
           "Every prompt below was run through the verified Qwen3.5-0.8B decode "
           "using int8 weights and int8 activations, the same arithmetic the FPGA "
           "datapath uses. The numpy golden model and the HLS decode agree bit for "
           "bit, so these are the exact tokens the hardware path produces. Timings "
           "are from the numpy reference on CPU and are for correctness, not speed. "
           "The real hardware token rate comes from the measured HBM bandwidth.\n"]

    for prompt, n_new, raw in PROMPTS:
        if raw:
            ids = tok(prompt, return_tensors="np").input_ids[0].tolist()
        else:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, tokenize=True)
            if not isinstance(enc, (list, tuple)):
                enc = enc["input_ids"]
            ids = np.array(enc).reshape(-1).tolist()

        st = DecodeState.new(cfg)
        t0 = time.perf_counter()
        logits = None
        for t in ids:
            logits = decode_step(W, cfg, int(t), st)
        t_pf = time.perf_counter() - t0

        gen, t_dec = [], 0.0
        for _ in range(n_new):
            nxt = int(np.argmax(logits))
            if nxt in eos:
                break
            gen.append(nxt)
            d0 = time.perf_counter()
            logits = decode_step(W, cfg, nxt, st)
            t_dec += time.perf_counter() - d0

        answer = tok.decode(gen)
        pf_rate = len(ids) / t_pf if t_pf else 0
        dc_rate = len(gen) / t_dec if t_dec else 0
        print(f"[{len(gen)} tok] {prompt[:50]}", flush=True)

        out.append(f"\n## {prompt}\n")
        out.append(f"```\n{answer}\n```")
        out.append(f"\n- prompt tokens: {len(ids)}")
        out.append(f"- generated tokens: {len(gen)}")
        out.append(f"- prefill: {t_pf:.2f}s ({pf_rate:.1f} tok/s)")
        out.append(f"- decode: {t_dec:.2f}s ({dc_rate:.1f} tok/s)\n")

    with open("artifacts/generations.md", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\nwrote artifacts/generations.md")


if __name__ == "__main__":
    main()
