// Gated full self-attention decode step (the 6 non-linear layers).
//
// Distinctive bits vs a vanilla attention: q_proj emits 2x head_dim per head
// (query + an output gate), QK-norm on the 256-dim head, partial RoPE over the
// first 64 dims, GQA (8 Q heads share 2 KV heads), and the output is gated by
// sigmoid(gate) before o_proj. Mirrors golden/model.py full_attention.
//
// KV cache is HBM-resident in hardware; here it is caller-owned flat buffers,
// laid out [token][NKV*HD]. Only 6 layers x 2 KV heads x 256 = 6 KB/token --
// the whole reason this hybrid model is a good FPGA target.

#pragma once
#include <cmath>

#include "gemv.hpp"
#include "norm.hpp"
#include "rope.hpp"
#include "types.hpp"

namespace qwen {

// out[HIDDEN]; x[HIDDEN] input-normed. pos = absolute position (for RoPE).
// kcache/vcache: [(T+1)*NKV*HD] caller buffers; on entry hold T past tokens,
// this step writes slot T. Attention runs over all T+1.
inline void attention_layer(float* out, const float* x,
                            const Tensor& q_w, const Tensor& k_w,
                            const Tensor& v_w, const Tensor& o_w,
                            const float* q_norm, const float* k_norm,
                            int pos, float* kcache, float* vcache, int T) {
  static float qg[Q_PROJ], kflat[KV_PROJ], vflat[KV_PROJ];
  gemv_i8(qg, q_w.w_i8, q_w.scale, x, Q_PROJ, HIDDEN);
  gemv_i8(kflat, k_w.w_i8, k_w.scale, x, KV_PROJ, HIDDEN);
  gemv_i8(vflat, v_w.w_i8, v_w.scale, x, KV_PROJ, HIDDEN);

  // split q / gate, QK-norm, RoPE
  static float q[NH][HD], gate[ATTN_O_IN];
  float cosb[ROPE_DIM], sinb[ROPE_DIM];
  rope_cos_sin(pos, cosb, sinb);
  for (int h = 0; h < NH; ++h) {
    rmsnorm(q[h], &qg[h * HD * 2], q_norm, HD, EPS);   // query = first HD of the 2*HD
    apply_rope(q[h], cosb, sinb);
    for (int d = 0; d < HD; ++d) gate[h * HD + d] = qg[h * HD * 2 + HD + d];
  }

  // K/V for this token -> cache slot T (K gets QK-norm + RoPE, V raw)
  float* kslot = kcache + (long)T * KV_PROJ;
  float* vslot = vcache + (long)T * KV_PROJ;
  for (int kv = 0; kv < NKV; ++kv) {
    float ktmp[HD];
    rmsnorm(ktmp, &kflat[kv * HD], k_norm, HD, EPS);
    apply_rope(ktmp, cosb, sinb);
    for (int d = 0; d < HD; ++d) kslot[kv * HD + d] = ktmp[d];
    for (int d = 0; d < HD; ++d) vslot[kv * HD + d] = vflat[kv * HD + d];
  }

  const int Tn = T + 1;                       // positions to attend over
  const int rep = NH / NKV;                   // 4
  const float scale = 1.0f / std::sqrt(float(HD));

  static float ctx[NH][HD];
ATTN_HEAD:
  for (int h = 0; h < NH; ++h) {
    const int kv = h / rep;
    // pass 1: scores + running max (online softmax would fuse these; two passes
    // is clearer and Tn is small for decode)
    static float sc[8192];                    // bounded context; raise if needed
    float m = -1e30f;
    for (int t = 0; t < Tn; ++t) {
      const float* kt = kcache + (long)t * KV_PROJ + kv * HD;
      float d = 0.0f;
      for (int i = 0; i < HD; ++i) d += q[h][i] * kt[i];
      d *= scale;
      sc[t] = d;
      if (d > m) m = d;
    }
    // pass 2: exp + normalise + weighted sum of V
    float denom = 0.0f;
    for (int t = 0; t < Tn; ++t) { sc[t] = std::exp(sc[t] - m); denom += sc[t]; }
    const float invd = 1.0f / denom;
    for (int d = 0; d < HD; ++d) ctx[h][d] = 0.0f;
    for (int t = 0; t < Tn; ++t) {
      const float w = sc[t] * invd;
      const float* vt = vcache + (long)t * KV_PROJ + kv * HD;
      for (int d = 0; d < HD; ++d) ctx[h][d] += w * vt[d];
    }
  }

  // flatten, output-gate, o_proj
  static float og[ATTN_O_IN];
  for (int h = 0; h < NH; ++h)
    for (int d = 0; d < HD; ++d)
      og[h * HD + d] = ctx[h][d] * sigmoidf(gate[h * HD + d]);

  gemv_i8(out, o_w.w_i8, o_w.scale, og, HIDDEN, ATTN_O_IN);
}

}  // namespace qwen
