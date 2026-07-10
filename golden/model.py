"""Pure-numpy fp32 decode of Qwen3.5-0.8B (text-only), token at a time.

This is the golden oracle for the FPGA build. Every HLS kernel is checked
against it. It is deliberately written as a *recurrent* decode -- one token
in, one logit vector out -- because that is exactly what the hardware does.
Prefill is just the same loop run over the prompt.

Transpiled from transformers.models.qwen3_5.modeling_qwen3_5 (v5.2.0):
  - torch_recurrent_gated_delta_rule    (line 403)
  - torch_causal_conv1d_update          (line 299)
  - Qwen3_5GatedDeltaNet.forward        (line 511)
  - Qwen3_5Attention.forward            (line 740)
  - Qwen3_5DecoderLayer.forward         (line 840)

Shapes for the 0.8B config:
  hidden 1024, 24 layers = [linear x3, full] x6
  linear: 16 kv heads, k/v head dim 128, conv k=4, conv_dim 6144
  full:   8 Q heads, 2 KV heads, head_dim 256, rotary dim 64, output-gated
  vocab 248320, tied embeddings
"""

from dataclasses import dataclass

import numpy as np

# Set to a dict to capture delta-rule inputs/outputs for HLS test vectors.
# {"layer": int, "step": int} -> filled with q,k,v,g,beta,S_in,o_core,S_out.
TRACE = None

# When True, quantise the ACTIVATION feeding each weight projection to int8
# (per-vector symmetric, dynamic). Models the int8-activation datapath the FPGA
# needs so the MAC is int8xint8 (2 per DSP) instead of fp32 (~5 DSP). Purely a
# quality probe -- see golden/eval_quant.py and docs/synthesis_estimate.py.
ACT_INT8 = False


def _qa(x):
    """Per-vector symmetric int8 round-trip of an activation, if ACT_INT8."""
    if not ACT_INT8:
        return x
    amax = np.abs(x).max()
    if amax == 0:
        return x
    s = amax / 127.0
    return np.rint(x / s).clip(-127, 127) * s

# ---------------------------------------------------------------- primitives


def silu(x):
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def softplus(x):
    # log1p(exp(x)) with the standard large-x guard; torch does the same.
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def rmsnorm(x, w, eps):
    """Qwen3_5RMSNorm -- note the (1 + w).

    The checkpoint stores gamma ZERO-CENTERED: `weight` is an offset from 1,
    not the gain itself (nn.Parameter(torch.zeros(dim)), then `* (1.0 + w)`).
    Used for input/post_attention layernorms, q_norm, k_norm, and final norm.

    Qwen3_5RMSNormGated (linear_attn.norm) does NOT do this -- it is
    ones-initialised and scales by `w` directly. See rmsnorm_gated below.
    Getting these two confused silently produces fluent garbage.
    """
    v = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(v + eps))) * (1.0 + w)


def l2norm(x, eps=1e-6):
    """FLA-compatible: rsqrt(sum(x^2) + eps). eps is INSIDE the sqrt."""
    return x * (1.0 / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps))


# ---------------------------------------------------------------- state


@dataclass
class DecodeState:
    conv: np.ndarray      # [n_linear, 6144, 4]      depthwise conv history
    rec: np.ndarray       # [n_linear, 16, 128, 128] DeltaNet recurrent state (fp32, ctx-independent)
    kc: list              # per full-attn layer: [T, 2, 256]
    vc: list
    pos: int = 0

    @staticmethod
    def new(cfg):
        lt = cfg["layer_types"]
        nlin = lt.count("linear_attention")
        nfull = lt.count("full_attention")
        cd = cfg["linear_key_head_dim"] * cfg["linear_num_key_heads"] * 2 \
            + cfg["linear_value_head_dim"] * cfg["linear_num_value_heads"]
        return DecodeState(
            conv=np.zeros((nlin, cd, cfg["linear_conv_kernel_dim"]), np.float32),
            rec=np.zeros((nlin, cfg["linear_num_value_heads"],
                          cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]), np.float32),
            kc=[np.zeros((0, cfg["num_key_value_heads"], cfg["head_dim"]), np.float32) for _ in range(nfull)],
            vc=[np.zeros((0, cfg["num_key_value_heads"], cfg["head_dim"]), np.float32) for _ in range(nfull)],
        )


# ---------------------------------------------------------------- rope


def rope_cos_sin(pos, head_dim, partial, theta):
    """Text-only: mRoPE's T/H/W sections are identical, so interleaving is a
    no-op and this collapses to plain partial RoPE."""
    dim = int(head_dim * partial)                       # 64
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))   # [32]
    f = pos * inv                                       # [32]
    emb = np.concatenate([f, f])                        # [64]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin):
    """x: [..., 256]. Rotate only the first 64 dims; pass the rest through."""
    rd = cos.shape[-1]                                  # 64
    rot, passthru = x[..., :rd], x[..., rd:]
    h = rd // 2
    half = np.concatenate([-rot[..., h:], rot[..., :h]], axis=-1)
    return np.concatenate([rot * cos + half * sin, passthru], axis=-1)


# ---------------------------------------------------------------- mixers


def gated_deltanet(W, p, cfg, x, st, li):
    """One decode step of a Gated DeltaNet layer. x: [1024] -> [1024]."""
    H = cfg["linear_num_value_heads"]        # 16
    K = cfg["linear_key_head_dim"]           # 128
    V = cfg["linear_value_head_dim"]         # 128
    kd = K * cfg["linear_num_key_heads"]     # 2048
    eps = cfg["rms_norm_eps"]

    xq = _qa(x)
    qkv = W[f"{p}.in_proj_qkv.weight"] @ xq                      # [6144]
    z = W[f"{p}.in_proj_z.weight"] @ xq                          # [2048]
    b = W[f"{p}.in_proj_b.weight"] @ xq                          # [16]
    a = W[f"{p}.in_proj_a.weight"] @ xq                          # [16]

    # depthwise causal conv1d, k=4, no bias, then SiLU.
    # torch keeps 4 columns of history but out[-1] only reads columns 1..4,
    # so column 0 is dead. Kept 4-wide to stay tensor-identical with torch.
    cw = W[f"{p}.conv1d.weight"][:, 0, :]                        # [6144, 4]
    xn = np.concatenate([st.conv[li], qkv[:, None]], axis=1)     # [6144, 5]
    st.conv[li] = xn[:, -4:]
    qkv = silu(np.sum(xn[:, 1:5] * cw, axis=1))                  # [6144]

    q = qkv[:kd].reshape(H, K)
    k = qkv[kd:2 * kd].reshape(H, K)
    v = qkv[2 * kd:].reshape(H, V)

    beta = sigmoid(b)                                            # [16]
    g = -np.exp(W[f"{p}.A_log"].astype(np.float32)) * softplus(a + W[f"{p}.dt_bias"])

    q = l2norm(q) * (K ** -0.5)                                  # scale AFTER l2norm
    k = l2norm(k)

    # --- the delta rule. S: [H, K, V], fp32, never leaves on-chip SRAM.
    S = st.rec[li]
    tr = TRACE if (TRACE is not None and TRACE.get("layer") == li) else None
    if tr is not None:
        tr.update(q=q.copy(), k=k.copy(), v=v.copy(), g=g.copy(),
                  beta=beta.copy(), S_in=S.copy())

    S *= np.exp(g)[:, None, None]                                # decay
    kv = np.einsum("hkv,hk->hv", S, k)                           # k^T S
    delta = (v - kv) * beta[:, None]
    S += np.einsum("hk,hv->hkv", k, delta)                       # rank-1 update
    o = np.einsum("hkv,hk->hv", S, q)                            # q^T S

    if tr is not None:
        tr.update(o_core=o.copy(), S_out=S.copy())

    # Qwen3_5RMSNormGated: normalise, scale by weight (NO 1+w here -- this one
    # is ones-initialised), THEN gate with silu(z).
    zz = z.reshape(H, V)
    var = np.mean(o.astype(np.float32) ** 2, axis=-1, keepdims=True)
    o = o * (1.0 / np.sqrt(var + eps))
    o = W[f"{p}.norm.weight"] * o
    o = o * silu(zz)

    return W[f"{p}.out_proj.weight"] @ _qa(o.reshape(-1))


def full_attention(W, p, cfg, x, st, fi, pos):
    """One decode step of a gated full-attention layer. x: [1024] -> [1024]."""
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]

    # q_proj emits 2x head_dim per head: the query and its output gate.
    xq = _qa(x)
    qg = (W[f"{p}.q_proj.weight"] @ xq).reshape(nh, hd * 2)
    q, gate = qg[:, :hd], qg[:, hd:]
    gate = gate.reshape(-1)                                      # [2048]

    q = rmsnorm(q, W[f"{p}.q_norm.weight"], eps)
    k = rmsnorm((W[f"{p}.k_proj.weight"] @ xq).reshape(nkv, hd), W[f"{p}.k_norm.weight"], eps)
    v = (W[f"{p}.v_proj.weight"] @ xq).reshape(nkv, hd)

    # torch reads both of these out of config.rope_parameters (compute_default_rope_parameters)
    rp = cfg["rope_parameters"]
    partial = rp.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
    cos, sin = rope_cos_sin(pos, hd, partial, rp["rope_theta"])
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    st.kc[fi] = np.concatenate([st.kc[fi], k[None]], axis=0)     # [T, 2, 256]
    st.vc[fi] = np.concatenate([st.vc[fi], v[None]], axis=0)

    rep = nh // nkv                                              # 4
    Kc = np.repeat(st.kc[fi], rep, axis=1)                       # [T, 8, 256]
    Vc = np.repeat(st.vc[fi], rep, axis=1)

    s = np.einsum("hd,thd->ht", q, Kc) * (hd ** -0.5)            # [8, T]
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s)
    w /= w.sum(-1, keepdims=True)
    o = np.einsum("ht,thd->hd", w, Vc).reshape(-1)               # [2048]

    o = o * sigmoid(gate)                                        # output gate
    return W[f"{p}.o_proj.weight"] @ _qa(o)


def mlp(W, p, x):
    xq = _qa(x)
    g = silu(W[f"{p}.gate_proj.weight"] @ xq)
    u = W[f"{p}.up_proj.weight"] @ xq
    return W[f"{p}.down_proj.weight"] @ _qa(g * u)


# ---------------------------------------------------------------- top level


def decode_step(W, cfg, tok, st):
    """One token in, [vocab] logits out. Mutates st."""
    eps = cfg["rms_norm_eps"]
    h = W["embed_tokens.weight"][tok].copy()
    li = fi = 0

    for i, kind in enumerate(cfg["layer_types"]):
        p = f"layers.{i}"
        r = h
        h = rmsnorm(h, W[f"{p}.input_layernorm.weight"], eps)
        if kind == "linear_attention":
            h = gated_deltanet(W, f"{p}.linear_attn", cfg, h, st, li)
            li += 1
        else:
            h = full_attention(W, f"{p}.self_attn", cfg, h, st, fi, st.pos)
            fi += 1
        h = r + h

        r = h
        h = rmsnorm(h, W[f"{p}.post_attention_layernorm.weight"], eps)
        h = r + mlp(W, f"{p}.mlp", h)

    st.pos += 1
    h = rmsnorm(h, W["norm.weight"], eps)
    return W["embed_tokens.weight"] @ h        # tied lm_head


def generate(W, cfg, prompt_ids, n_new=16, greedy=True):
    st = DecodeState.new(cfg)
    logits = None
    for t in prompt_ids:
        logits = decode_step(W, cfg, int(t), st)
    out = []
    for _ in range(n_new):
        nxt = int(np.argmax(logits)) if greedy else int(np.argmax(logits))
        out.append(nxt)
        logits = decode_step(W, cfg, nxt, st)
    return out, st
