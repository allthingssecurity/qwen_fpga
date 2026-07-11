"""Export a real partial-RoPE case: a query head after QK-norm, the cos/sin for
its position, and the golden rotated output. Only the first 64 of 256 dims rotate.

  python3 scripts/export_rope_tv.py  ->  artifacts/tv_rope.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import apply_rope, rmsnorm, rope_cos_sin  # noqa: E402


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)
    p = "layers.3.self_attn"
    hd = cfg["head_dim"]                       # 256

    x = rmsnorm(W["embed_tokens.weight"][6511].astype(np.float32),
                W["layers.3.post_attention_layernorm.weight"], cfg["rms_norm_eps"])
    qg = (W[f"{p}.q_proj.weight"] @ x).reshape(cfg["num_attention_heads"], hd * 2)
    q0 = qg[0, :hd]                            # head 0 query
    q0n = rmsnorm(q0, W[f"{p}.q_norm.weight"], cfg["rms_norm_eps"])
    rp = cfg["rope_parameters"]
    cos, sin = rope_cos_sin(5, hd, rp.get("partial_rotary_factor", 0.25), rp["rope_theta"])
    out = apply_rope(q0n, cos, sin)

    print(f"HD={hd} ROPE_DIM={cos.size}  q range [{q0n.min():.2f},{q0n.max():.2f}]  "
          f"out range [{out.min():.2f},{out.max():.2f}]")
    with open("artifacts/tv_rope.bin", "wb") as f:
        np.array([hd, cos.size], "<i4").tofile(f)
        q0n.astype("<f4").tofile(f)
        cos.astype("<f4").tofile(f)
        sin.astype("<f4").tofile(f)
        out.astype("<f4").tofile(f)
    print("wrote artifacts/tv_rope.bin")


if __name__ == "__main__":
    main()
