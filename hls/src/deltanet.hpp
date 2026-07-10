// Gated DeltaNet recurrent step -- the FPGA-side crux of Qwen3.5-0.8B decode.
//
// State S[H][K][V] = 16 x 128 x 128 fp32 = 1 MiB per layer, 18 MiB across all
// 18 linear layers. That fits in VU47P on-chip URAM (270 Mb = 33.75 MiB), so
// the recurrent state NEVER touches HBM and never grows with context length.
// This is the whole reason the hybrid model is a better FPGA target than a
// dense one: a dense 0.6B pays 56 KB/token of KV traffic, this pays 6 KB.
//
// fp32 is not a choice -- config.json says mamba_ssm_dtype: float32. A
// recurrent state in bf16 accumulates error across every token forever.
//
// Per token per layer, exactly two sweeps over S:
//   A:  S[k][v] *= exp(g);          kv[v]  += S[k][v] * key[k]
//       delta[v] = (val[v] - kv[v]) * beta
//   B:  S[k][v] += key[k]*delta[v]; out[v] += S[k][v] * qry[k]
//
// Sweep B must read the UPDATED S -- out = q^T S_after, not S_before. Getting
// that backwards still produces fluent text, which is the trap.
//
// Cost: 2 * 128 * 128 MAC per head = 32,768; x16 heads x18 layers = 9.4 MMAC
// per token. At 250 MHz with V unrolled 128-wide that is ~4,096 cycles/layer,
// 73,728 cycles/token = 295 us => ~3,390 tok/s of DeltaNet headroom, sitting
// comfortably under the 1.94 ms/token (515 tok/s) HBM weight-streaming budget.
// The state math hides entirely under the weight stream. HBM is the critical
// path; this block must merely not become one.
//
// Compiles two ways from one source:
//   clang++ -Wno-unknown-pragmas   -> csim, bit-checked against numpy golden
//   Vitis HLS                      -> pragmas live, synthesises to the VU47P

#pragma once

#include <cmath>

#include "activations.hpp"
#include "conv1d.hpp"
#include "gemv.hpp"
#include "norm.hpp"
#include "types.hpp"

namespace qwen {

constexpr int H = 16;   // linear_num_value_heads  (== LH)
constexpr int K = 128;  // linear_key_head_dim     (== LK)
constexpr int V = 128;  // linear_value_head_dim   (== LV)

// Lanes across the value dim. NOT full 128: at 128-wide + II=1 the fp32 MAC in
// the kv[v]/out[v] recurrence is forced combinational (recurrence distance = 1
// cycle < fp-add latency), giving a ~15 ns critical path (~66 MHz) -- the exact
// failure Vitis reported. Tiling the value dim into DN_LANES-wide chunks makes
// the recurrence distance V/DN_LANES cycles, so HLS can REGISTER the fp ops and
// still pipeline at II=1. 16 lanes -> distance 8 cycles > fp-add latency, and
// 128*(128/16)=1024 cycles/sweep => ~424 tok/s, comfortably under HBM budget.
// This trades DeltaNet's large time slack for timing closure. Re-synth confirms.
constexpr int DN_LANES = 16;

// One head. S is updated in place; out is written.
inline void delta_rule_head(const float qry[K], const float key[K], const float val[V],
                            float g, float beta, float S[K][V], float out[V]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = S dim = 2 cyclic factor = DN_LANES
#pragma HLS ARRAY_PARTITION variable = out dim = 1 cyclic factor = DN_LANES

  const float gexp = std::exp(g);  // g <= 0, so gexp in (0,1] -- a decay

  float kv[V];
#pragma HLS ARRAY_PARTITION variable = kv dim = 1 cyclic factor = DN_LANES
SWEEP_A_INIT:
  for (int v = 0; v < V; ++v) {
#pragma HLS UNROLL factor = DN_LANES
    kv[v] = 0.0f;
  }

  // --- sweep A: decay in place, and contract k^T S into kv (tiled over v)
SWEEP_A:
  for (int k = 0; k < K; ++k) {
    const float kk = key[k];
    for (int vt = 0; vt < V; vt += DN_LANES) {
#pragma HLS PIPELINE II = 1
      for (int j = 0; j < DN_LANES; ++j) {
#pragma HLS UNROLL
        const int v = vt + j;
        const float s = S[k][v] * gexp;
        S[k][v] = s;
        kv[v] += s * kk;
      }
    }
  }

  float delta[V];
#pragma HLS ARRAY_PARTITION variable = delta dim = 1 cyclic factor = DN_LANES
DELTA:
  for (int v = 0; v < V; ++v) {
#pragma HLS UNROLL factor = DN_LANES
    delta[v] = (val[v] - kv[v]) * beta;
    out[v] = 0.0f;
  }

  // --- sweep B: rank-1 update, then contract q^T S_after into out (tiled over v)
SWEEP_B:
  for (int k = 0; k < K; ++k) {
    const float kk = key[k];
    const float qq = qry[k];
    for (int vt = 0; vt < V; vt += DN_LANES) {
#pragma HLS PIPELINE II = 1
      for (int j = 0; j < DN_LANES; ++j) {
#pragma HLS UNROLL
        const int v = vt + j;
        const float s = S[k][v] + kk * delta[v];
        S[k][v] = s;
        out[v] += s * qq;  // uses updated s -- deliberate
      }
    }
  }
}

// Full 16-head layer step. Heads are independent: unroll or pipeline at will.
inline void delta_rule_layer(const float qry[H][K], const float key[H][K], const float val[H][V],
                             const float g[H], const float beta[H],
                             float S[H][K][V], float out[H][V]) {
#pragma HLS ARRAY_PARTITION variable = S dim = 1 cyclic factor = 4
HEADS:
  for (int h = 0; h < H; ++h) {
#pragma HLS LOOP_FLATTEN off
    delta_rule_head(qry[h], key[h], val[h], g[h], beta[h], S[h], out[h]);
  }
}

// l2 normalise over a head dim: rsqrt(sum(x^2) + eps), eps INSIDE the sqrt (FLA
// convention). Matches golden l2norm.
inline void l2norm(float* v, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) ss += v[i] * v[i];
  const float inv = 1.0f / std::sqrt(ss + 1e-6f);
  for (int i = 0; i < n; ++i) v[i] *= inv;
}

// Full Gated DeltaNet layer decode step. x is the input-normed hidden [HIDDEN];
// out is [HIDDEN]. rec/conv_state persist across tokens (on-chip URAM in HW).
// Mirrors golden/model.py gated_deltanet exactly.
inline void deltanet_layer(float* out, const float* x,
                           const Tensor& qkv_w, const Tensor& z_w,
                           const Tensor& b_w, const Tensor& a_w,
                           const float* conv_w, const float* A_log,
                           const float* dt_bias, const float* gate_norm_w,
                           const Tensor& out_w,
                           float rec[H][K][V], float conv_state[CONV_DIM][CONV_K]) {
  static float qkv[QKV_DIM], z[Z_DIM], b[LH], a[LH];
  gemv_i8(qkv, qkv_w.w_i8, qkv_w.scale, x, QKV_DIM, HIDDEN);
  gemv_i8(z, z_w.w_i8, z_w.scale, x, Z_DIM, HIDDEN);
  gemv_f32(b, b_w.w_f32, x, LH, HIDDEN);
  gemv_f32(a, a_w.w_f32, x, LH, HIDDEN);

  // short conv + SiLU over the 6144-wide qkv stream
  static float convd[CONV_DIM];
  conv1d_step(convd, qkv, conv_state, conv_w);

  // split into heads and gates
  static float q[H][K], k[H][K], v[H][V], g[LH], beta[LH];
  for (int h = 0; h < H; ++h) {
    for (int i = 0; i < K; ++i) q[h][i] = convd[h * K + i];
    for (int i = 0; i < K; ++i) k[h][i] = convd[VALUE_DIM + h * K + i];
    for (int i = 0; i < V; ++i) v[h][i] = convd[2 * VALUE_DIM + h * V + i];
    beta[h] = sigmoidf(b[h]);
    g[h] = -std::exp(A_log[h]) * softplusf(a[h] + dt_bias[h]);
    l2norm(q[h], K);
    l2norm(k[h], K);
    const float scale = 1.0f / std::sqrt(float(K));
    for (int i = 0; i < K; ++i) q[h][i] *= scale;
  }

  static float o[H][V];
  delta_rule_layer(q, k, v, g, beta, rec, o);

  // per-head gated RMSNorm, gated by silu(z_head)
  static float og[VALUE_DIM];
  for (int h = 0; h < H; ++h) {
    rmsnorm_gated(&og[h * V], o[h], gate_norm_w, &z[h * V], V, EPS);
  }

  gemv_i8(out, out_w.w_i8, out_w.scale, og, HIDDEN, VALUE_DIM);
}

}  // namespace qwen
