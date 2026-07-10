#!/usr/bin/env python3
"""Run YOUR prompt through Qwen3.5-0.8B and print the answer + speed numbers.

This uses the numpy golden model -- the bit-exact reference, not the FPGA. So
the *speed* you see here is "fp32 numpy on your CPU", which is slow and means
nothing on its own. Its job is to produce the real tokens. Alongside it we
print the PROJECTED F2 FPGA decode rate from the validated HBM bandwidth model
(docs/hbm_bandwidth.md), which is the number that actually matters.

    python3 run_prompt.py -p "Explain FPGAs to a five year old." -n 80
    python3 run_prompt.py -p "def fib(n):" --raw          # raw completion, no chat wrap
    python3 run_prompt.py -p "hi" --int8                   # run the int8 weights (FPGA-accurate quality)

No torch needed -- numpy + the HF tokenizer only.
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "golden")
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import DecodeState, decode_step  # noqa: E402
from quant import quantize_weights  # noqa: E402

# validated F2 device ceiling: channel-interleaved layout, depth 384 (see docs/)
F2_TOK_S = 559.0
F2_GBPS = 424.0
MB_PER_TOKEN = 758.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-n", "--n_new", type=int, default=64, help="max tokens to generate")
    ap.add_argument("--int8", action="store_true", help="use int8 weights (what the FPGA runs)")
    ap.add_argument("--w8a8", action="store_true",
                    help="int8 weights AND int8 activations -- the exact FPGA datapath (94.8%%)")
    ap.add_argument("--raw", action="store_true", help="raw completion; skip the chat template")
    args = ap.parse_args()

    import model as _m
    _m.ACT_INT8 = args.w8a8

    snap = find_snapshot()
    cfg = load_config(snap)
    print("loading weights (fp32)...", flush=True)
    W = load_text_weights(snap, verbose=False)
    if args.int8 or args.w8a8:
        print("fake-quantising weights to int8...", flush=True)
        W = quantize_weights(W, exclude_embed=False, verbose=False)[0]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)

    if args.raw:
        ids = tok(args.prompt, return_tensors="np").input_ids[0].tolist()
    else:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, tokenize=True,
        )
        if not isinstance(enc, (list, tuple, np.ndarray)):   # transformers 5.x: BatchEncoding
            enc = enc["input_ids"]
        ids = np.array(enc).reshape(-1).tolist()
    eos = {tok.eos_token_id, cfg.get("eos_token_id")}
    eos.discard(None)

    dp = "W8A8 (exact FPGA datapath)" if args.w8a8 else \
         ("int8 weights / fp32 act" if args.int8 else "fp32 (reference)")
    print(f"\nprompt      : {args.prompt!r}")
    print(f"weights     : {dp}")
    print(f"prompt len  : {len(ids)} tokens\n")

    st = DecodeState.new(cfg)

    # --- prefill: feed the prompt
    t0 = time.perf_counter()
    logits = None
    for t in ids:
        logits = decode_step(W, cfg, int(t), st)
    t_prefill = time.perf_counter() - t0

    # --- decode: greedy, stop on EOS
    print("answer      : ", end="", flush=True)
    out, t_dec = [], 0.0
    for _ in range(args.n_new):
        nxt = int(np.argmax(logits))
        if nxt in eos:
            break
        out.append(nxt)
        # incremental detokenise for a live feel
        print(tok.decode([nxt]), end="", flush=True)
        d0 = time.perf_counter()
        logits = decode_step(W, cfg, nxt, st)
        t_dec += time.perf_counter() - d0
    print("\n")

    n_gen = len(out)
    pf_rate = len(ids) / t_prefill if t_prefill else 0
    dc_rate = n_gen / t_dec if t_dec else 0

    print("--- speed ------------------------------------------------")
    print(f"prefill     : {len(ids):>4} tok in {t_prefill:6.2f}s   {pf_rate:7.1f} tok/s   (numpy/CPU)")
    print(f"decode      : {n_gen:>4} tok in {t_dec:6.2f}s   {dc_rate:7.1f} tok/s   (numpy/CPU, reference only)")
    print("----------------------------------------------------------")
    print(f"PROJECTED on AWS F2 FPGA (validated HBM model):")
    print(f"  decode    : {F2_TOK_S:7.0f} tok/s   ({F2_GBPS:.0f} GB/s, 92% of 460 GB/s peak)")
    print(f"  this answer ({n_gen} tok) would take {n_gen / F2_TOK_S * 1000:.0f} ms  "
          f"vs {t_dec:.1f}s here  ({t_dec / (n_gen / F2_TOK_S):.0f}x faster)")
    print(f"  (device ceiling; F2 shell/AXI derates by an unmeasured factor)")


if __name__ == "__main__":
    main()
