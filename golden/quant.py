"""Per-output-channel symmetric int8 weight quantisation.

Decode is bandwidth-bound, so *only weight bytes matter*. Activations are a
1024-vector -- quantising them buys ~nothing on bandwidth and costs accuracy,
so we leave them fp32. (Fixed-point activations are a DSP-cost question for
the RTL, not a bandwidth question. Separate decision, separate experiment.)

What we do NOT quantise, and why:
  norms, A_log, dt_bias   -- a few hundred bytes; feed exp()/softplus() and set
                             the DeltaNet decay. Quantising them perturbs the
                             recurrence itself.
  in_proj_a / in_proj_b   -- [16,1024] each, 16K params. Produce g and beta,
                             which go through exp() and sigmoid(). Sensitive,
                             and free to keep in fp32.
  conv1d.weight           -- [6144,4], 24.5K params. Negligible.

Everything on QUANT_SUFFIXES is a real GEMV weight and carries the bandwidth.
"""

import numpy as np

QUANT_SUFFIXES = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "embed_tokens.weight",
)


def should_quant(name: str, exclude_embed: bool = False) -> bool:
    if exclude_embed and name == "embed_tokens.weight":
        return False
    return any(name.endswith(s) for s in QUANT_SUFFIXES)


def quant_rows(w: np.ndarray):
    """W[out,in] -> (int8 q[out,in], fp32 scale[out]). Symmetric, per output row."""
    assert w.ndim == 2
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def dequant_rows(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale[:, None]


def quantize_weights(W: dict, exclude_embed: bool = False, verbose: bool = True):
    """Return a fake-quantised copy (dequantised back to fp32) plus a byte ledger."""
    out, ledger = {}, {"q_params": 0, "q_bytes": 0, "fp32_params": 0, "fp32_bytes": 0, "scale_bytes": 0}
    for k, v in W.items():
        if should_quant(k, exclude_embed) and v.ndim == 2:
            q, s = quant_rows(v)
            out[k] = dequant_rows(q, s)
            ledger["q_params"] += v.size
            ledger["q_bytes"] += q.nbytes
            ledger["scale_bytes"] += s.nbytes
        else:
            out[k] = v
            ledger["fp32_params"] += v.size
            ledger["fp32_bytes"] += v.nbytes
    if verbose:
        tot = ledger["q_bytes"] + ledger["scale_bytes"] + ledger["fp32_bytes"]
        print(f"  int8 : {ledger['q_params']:>12,} params -> {ledger['q_bytes']/1e6:8.1f} MB "
              f"(+{ledger['scale_bytes']/1e6:.2f} MB scales)")
        print(f"  fp32 : {ledger['fp32_params']:>12,} params -> {ledger['fp32_bytes']/1e6:8.1f} MB")
        print(f"  TOTAL bytes read per decode token: {tot/1e6:.1f} MB")
    return out, ledger


def rel_err(W, Wq):
    """Per-tensor relative Frobenius error, worst first."""
    rows = []
    for k in W:
        if W[k] is not Wq[k] and W[k].shape == Wq[k].shape and not np.shares_memory(W[k], Wq[k]):
            d = np.linalg.norm(W[k] - Wq[k]) / (np.linalg.norm(W[k]) + 1e-12)
            if d > 0:
                rows.append((d, k))
    rows.sort(reverse=True)
    return rows
