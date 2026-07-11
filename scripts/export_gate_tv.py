"""Export the DeltaNet gate math for a real token:
   g    = -exp(A_log) * softplus(a + dt_bias)     (a = in_proj_a @ x)
   gexp = exp(g)                                   (feeds the recurrence decay)
   beta = sigmoid(b)                               (b = in_proj_b @ x)
Also exports the softplus, exp, and sigmoid lookup tables.

  python3 scripts/export_gate_tv.py  ->  artifacts/tv_gate.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from loader import find_snapshot, load_config, load_text_weights  # noqa: E402
from model import rmsnorm, sigmoid, softplus  # noqa: E402

LUTN = 1024


def main():
    snap = find_snapshot()
    cfg = load_config(snap)
    W = load_text_weights(snap, verbose=False)
    p = "layers.0.linear_attn"
    x = rmsnorm(W["embed_tokens.weight"][11751].astype(np.float32),
                W["layers.0.input_layernorm.weight"], cfg["rms_norm_eps"])
    a = (W[f"{p}.in_proj_a.weight"] @ x).astype(np.float32)     # [16]
    b = (W[f"{p}.in_proj_b.weight"] @ x).astype(np.float32)     # [16]
    A = np.exp(W[f"{p}.A_log"].astype(np.float32))              # exp(A_log)
    dt = W[f"{p}.dt_bias"].astype(np.float32)

    g = (-A * softplus(a + dt)).astype(np.float32)
    gexp = np.exp(g).astype(np.float32)
    beta = sigmoid(b).astype(np.float32)

    # LUTs sampled at bin LEFT EDGES so linear interpolation in the RTL is exact
    # (no half-bin phase error). softplus [-16,16], exp [-16,0], sigmoid [-16,16].
    xs_sp = np.linspace(-16, 16, LUTN, endpoint=False)
    xs_ex = np.linspace(-16, 0, LUTN, endpoint=False)
    xs_sg = np.linspace(-16, 16, LUTN, endpoint=False)
    lut_sp = softplus(xs_sp).astype(np.float32)
    lut_ex = np.exp(xs_ex).astype(np.float32)
    lut_sg = sigmoid(xs_sg).astype(np.float32)

    print(f"a+dt range [{(a+dt).min():.2f},{(a+dt).max():.2f}]  A range [{A.min():.2f},{A.max():.2f}]")
    print(f"g range [{g.min():.2f},{g.max():.2f}]  gexp [{gexp.min():.4f},{gexp.max():.4f}]  "
          f"beta [{beta.min():.3f},{beta.max():.3f}]")
    with open("artifacts/tv_gate.bin", "wb") as f:
        np.array([16, LUTN], "<i4").tofile(f)
        lut_sp.tofile(f); lut_ex.tofile(f); lut_sg.tofile(f)
        a.tofile(f); b.tofile(f); A.tofile(f); dt.tofile(f)
        gexp.tofile(f); beta.tofile(f)
    print("wrote artifacts/tv_gate.bin")


if __name__ == "__main__":
    main()
