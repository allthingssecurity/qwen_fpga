"""Pack Qwen3.5-0.8B text LM into an HBM-resident int8 blob + JSON manifest.

Layout is chosen for the decode dataflow, not for tidiness: tensors are ordered
exactly in the sequence the kernel reads them (layer 0 .. 23, then lm_head), so
one decode token is a single forward sweep through HBM with no seeking. Each
tensor is 4 KiB-aligned so every AXI burst starts on a page boundary.

  int8 tensor:  [out, in] row-major int8   then  [out] fp32 per-row scales
  fp32 tensor:  as-is

  python3 scripts/pack_weights.py --out artifacts/qwen35_int8.bin
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from quant import quant_rows, should_quant  # noqa: E402

ALIGN = 4096


def read_order(cfg):
    """The exact order the decode kernel touches weights."""
    order = []
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"layers.{i}"
        order.append(f"{p}.input_layernorm.weight")
        if kind == "linear_attention":
            a = f"{p}.linear_attn"
            order += [f"{a}.in_proj_qkv.weight", f"{a}.in_proj_z.weight",
                      f"{a}.in_proj_b.weight", f"{a}.in_proj_a.weight",
                      f"{a}.conv1d.weight", f"{a}.A_log", f"{a}.dt_bias",
                      f"{a}.norm.weight", f"{a}.out_proj.weight"]
        else:
            a = f"{p}.self_attn"
            order += [f"{a}.q_proj.weight", f"{a}.k_proj.weight", f"{a}.v_proj.weight",
                      f"{a}.q_norm.weight", f"{a}.k_norm.weight", f"{a}.o_proj.weight"]
        order.append(f"{p}.post_attention_layernorm.weight")
        order += [f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight", f"{p}.mlp.down_proj.weight"]
    order += ["norm.weight", "embed_tokens.weight"]
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/qwen35_int8.bin")
    ap.add_argument("--check", action="store_true", help="re-read and verify round-trip")
    args = ap.parse_args()

    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap)

    order = read_order(cfg)
    missing = [k for k in order if k not in W]
    extra = [k for k in W if k not in order]
    if missing or extra:
        raise SystemExit(f"manifest/weights mismatch\n  missing: {missing[:5]}\n  extra: {extra[:5]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest, off = [], 0
    with open(args.out, "wb") as f:
        for name in order:
            v = W[name]
            pad = (-off) % ALIGN
            if pad:
                f.write(b"\0" * pad)
                off += pad
            ent = {"name": name, "shape": list(v.shape), "offset": off}
            if should_quant(name) and v.ndim == 2:
                q, s = quant_rows(v)
                f.write(q.tobytes())
                ent["dtype"] = "int8"
                ent["nbytes"] = q.nbytes
                ent["scale_offset"] = off + q.nbytes
                ent["scale_nbytes"] = s.nbytes
                f.write(s.astype("<f4").tobytes())
                off += q.nbytes + s.nbytes
            else:
                f.write(v.astype("<f4").tobytes())
                ent["dtype"] = "fp32"
                ent["nbytes"] = v.astype("<f4").nbytes
                off += ent["nbytes"]
            manifest.append(ent)

    total = off
    mpath = os.path.splitext(args.out)[0] + ".manifest.json"
    with open(mpath, "w") as f:
        json.dump({"config": {k: cfg[k] for k in
                              ("hidden_size", "num_hidden_layers", "layer_types", "head_dim",
                               "num_attention_heads", "num_key_value_heads", "intermediate_size",
                               "vocab_size", "rms_norm_eps", "linear_num_value_heads",
                               "linear_key_head_dim", "linear_value_head_dim",
                               "linear_conv_kernel_dim", "rope_parameters")},
                   "align": ALIGN, "total_bytes": total, "tensors": manifest}, f, indent=1)

    q_b = sum(e["nbytes"] for e in manifest if e["dtype"] == "int8")
    s_b = sum(e.get("scale_nbytes", 0) for e in manifest)
    f_b = sum(e["nbytes"] for e in manifest if e["dtype"] == "fp32")
    print(f"wrote {args.out}")
    print(f"  int8 payload {q_b/1e6:8.1f} MB")
    print(f"  fp32 scales  {s_b/1e6:8.2f} MB")
    print(f"  fp32 tensors {f_b/1e6:8.2f} MB")
    print(f"  padding      {(total-q_b-s_b-f_b)/1e6:8.2f} MB")
    print(f"  TOTAL        {total/1e6:8.1f} MB   ({total/2**20:.0f} MiB)")
    print(f"  fits 16 GiB HBM: {'yes' if total < 16*2**30 else 'NO'}   "
          f"roofline @391 GB/s: {391e9/total:.0f} tok/s")
    print(f"wrote {mpath}")

    if args.check:
        print("\nround-trip check vs in-memory fake-quant...")
        mm = np.memmap(args.out, dtype=np.uint8, mode="r")
        worst = 0.0
        for e in manifest:
            n = e["name"]
            if e["dtype"] == "int8":
                q = np.frombuffer(mm[e["offset"]:e["offset"] + e["nbytes"]].tobytes(),
                                  dtype=np.int8).reshape(e["shape"])
                s = np.frombuffer(mm[e["scale_offset"]:e["scale_offset"] + e["scale_nbytes"]].tobytes(),
                                  dtype="<f4")
                got = q.astype(np.float32) * s[:, None]
                qq, ss = quant_rows(W[n])
                exp = qq.astype(np.float32) * ss[:, None]
            else:
                got = np.frombuffer(mm[e["offset"]:e["offset"] + e["nbytes"]].tobytes(),
                                    dtype="<f4").reshape(e["shape"])
                exp = W[n]
            d = float(np.abs(got - exp).max())
            worst = max(worst, d)
        print(f"  worst |packed - fakequant| = {worst:.3e}  {'OK' if worst == 0.0 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
