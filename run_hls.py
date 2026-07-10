#!/usr/bin/env python3
"""Run YOUR prompt through the synthesisable HLS decode datapath.

Unlike run_prompt.py (numpy reference), this drives the actual HLS C++ kernels
(qwen_synth.hpp) -- the same source that goes to Vitis -- reading int8 weights
from the packed HBM blob. It's the closest you can get to "running it on the
FPGA" without an FPGA: identical arithmetic, identical weight layout, just
executed on the CPU instead of the fabric.

    python3 run_hls.py -p "Explain what an FPGA is." -n 40

Requires the packed blob + offset header (auto-built on first run).
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "artifacts", "run_hls")
BLOB = os.path.join(HERE, "artifacts", "qwen35_int8.bin")
F2_TOK_S = 559.0


def sh(cmd):
    subprocess.run(cmd, check=True, cwd=HERE)


def ensure_built():
    if not os.path.exists(BLOB):
        print("packing int8 weights (one-time)...", flush=True)
        sh([sys.executable, "scripts/pack_weights.py"])
    if not os.path.exists(os.path.join(HERE, "hls/src/weight_offsets.hpp")):
        sh([sys.executable, "scripts/gen_offsets.py"])
    # (re)build the driver
    sh(["make", "-C", "hls", "run_hls"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-n", "--n_new", type=int, default=40)
    ap.add_argument("--raw", action="store_true", help="raw completion; skip chat template")
    args = ap.parse_args()

    ensure_built()

    sys.path.insert(0, os.path.join(HERE, "golden"))
    from loader import find_snapshot, load_config
    from transformers import AutoTokenizer

    snap = find_snapshot()
    cfg = load_config(snap)
    tok = AutoTokenizer.from_pretrained(snap)

    if args.raw:
        ids = tok(args.prompt, return_tensors="np").input_ids[0].tolist()
    else:
        enc = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                      add_generation_prompt=True, tokenize=True)
        if not isinstance(enc, (list, tuple)):
            enc = enc["input_ids"]
        import numpy as np
        ids = np.array(enc).reshape(-1).tolist()

    eos = tok.eos_token_id if tok.eos_token_id is not None else -1
    idsfile = os.path.join(HERE, "artifacts", "prompt_ids.txt")
    with open(idsfile, "w") as f:
        f.write(" ".join(str(int(i)) for i in ids))

    print(f"\nprompt   : {args.prompt!r}  ({len(ids)} tokens)")
    print(f"datapath : synthesisable HLS kernels (qwen_synth.hpp), int8 weights\n")
    print("answer   : ", end="", flush=True)

    t0 = time.perf_counter()
    out = subprocess.run([BIN, BLOB, idsfile, str(args.n_new), str(eos)],
                         cwd=HERE, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if out.returncode != 0:
        print("\n[driver error]\n" + out.stderr)
        return 1

    gen_ids = [int(x) for x in out.stdout.split()] if out.stdout.strip() else []
    print(tok.decode(gen_ids))
    print()
    sys.stderr.write(out.stderr)
    n = len(gen_ids)
    print(f"\n--- this ran on the CPU via csim (the HLS datapath, not the FPGA).")
    print(f"    projected on AWS F2: {F2_TOK_S:.0f} tok/s -> {n} tok in {n / F2_TOK_S * 1000:.0f} ms")


if __name__ == "__main__":
    main()
