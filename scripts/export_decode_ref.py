"""Golden INT8 decode reference, for the end-to-end HLS testbench.

Runs the numpy golden model with the SAME int8 fake-quant the FPGA uses, over
the same fed_tokens as artifacts/ref.npz, and dumps per-step argmax + logits.
The HLS decode (which reads the identical packed int8 weights) must reproduce
these. Comparing against the int8 golden -- not the fp32 one -- isolates the
hardware datapath from quantisation error: any gap is fp summation ORDER only.

  python3 scripts/export_decode_ref.py   ->  artifacts/decode_ref.npz
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import DecodeState, decode_step  # noqa: E402
from quant import quantize_weights  # noqa: E402


def main():
    ref = np.load("artifacts/ref.npz")
    fed = ref["fed_tokens"].tolist()
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap)
    print("fake-quantising to int8 (same as the packed blob)...")
    Wq = quantize_weights(W, exclude_embed=False, verbose=False)[0]

    st = DecodeState.new(cfg)
    argmax, logit_rows = [], []
    for t in fed:
        lg = decode_step(Wq, cfg, int(t), st)
        argmax.append(int(np.argmax(lg)))
        logit_rows.append(lg.astype(np.float32))

    out = {
        "fed_tokens": np.array(fed, np.int64),
        "argmax": np.array(argmax, np.int64),
        "logits": np.stack(logit_rows),          # [steps, VOCAB]
        "prompt_len": ref["prompt_len"],
    }
    np.savez("artifacts/decode_ref.npz", **out)

    # flat binary the C++ testbench reads: n, fed[n], argmax[n], logits_step0[VOCAB]
    with open("artifacts/decode_ref.bin", "wb") as f:
        n = len(fed)
        np.array([n], np.int32).tofile(f)
        np.array(fed, np.int32).tofile(f)
        np.array(argmax, np.int32).tofile(f)
        logit_rows[0].astype("<f4").tofile(f)   # step-0 logits for a numeric check

    print(f"wrote artifacts/decode_ref.npz + .bin  ({len(fed)} steps)")
    print(f"  int8 argmax stream: {argmax}")


if __name__ == "__main__":
    main()
