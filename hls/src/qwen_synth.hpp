// SYNTHESISABLE decode top. Same datapath as decoder.hpp, but weights are
// addressed through the compile-time offset tables (weight_offsets.hpp) instead
// of a std::string map -- so this version actually goes through Vitis HLS.
//
// Verified by tb_decode alongside the string-based reference: both must match
// the golden int8 model at every step. If they diverge, the offset table is
// wrong; if they agree, the synthesisable addressing is correct.

#pragma once
#include "attention.hpp"
#include "deltanet.hpp"
#include "gemv.hpp"
#include "mlp.hpp"
#include "norm.hpp"
#include "state.hpp"
#include "types.hpp"
#include "weight_offsets.hpp"

namespace qwen {

inline Tensor tensor_of(const uint8_t* base, const TOff& o) {
  Tensor t;
  t.out = o.out;
  t.in = o.in;
  if (o.dtype == 1) {
    t.w_i8 = reinterpret_cast<const int8_t*>(base + o.off);
    t.scale = reinterpret_cast<const float*>(base + o.soff);
  } else {
    t.w_f32 = reinterpret_cast<const float*>(base + o.off);
  }
  return t;
}
inline const float* f32_of(const uint8_t* base, const TOff& o) {
  return reinterpret_cast<const float*>(base + o.off);
}

// One decode token, reading all weights from the packed HBM blob `hbm`.
// Returns argmax; fills logits[VOCAB] if non-null.
inline int qwen_top(const uint8_t* hbm, DecodeState& st, int tok, float* logits) {
  static float h[HIDDEN], hn[HIDDEN], mix[HIDDEN], r[HIDDEN];

  const Tensor emb = tensor_of(hbm, W_EMBED);
  embed_row(h, emb.w_i8, emb.scale, tok, HIDDEN);

  int li = 0, fi = 0;
LAYERS:
  for (int layer = 0; layer < N_LAYERS; ++layer) {
    for (int i = 0; i < HIDDEN; ++i) r[i] = h[i];
    rmsnorm(hn, h, f32_of(hbm, W_IN_LN[layer]), HIDDEN, EPS);

    if (is_linear(layer)) {
      deltanet_layer(mix, hn,
                     tensor_of(hbm, W_QKV[layer]), tensor_of(hbm, W_Z[layer]),
                     tensor_of(hbm, W_B[layer]), tensor_of(hbm, W_A[layer]),
                     f32_of(hbm, W_CONV[layer]), f32_of(hbm, W_ALOG[layer]),
                     f32_of(hbm, W_DTB[layer]), f32_of(hbm, W_LNORM[layer]),
                     tensor_of(hbm, W_OUTP[layer]),
                     st.rec[li], st.conv[li]);
      ++li;
    } else {
      st.kc[fi].resize((long)(st.pos + 1) * KV_PROJ);
      st.vc[fi].resize((long)(st.pos + 1) * KV_PROJ);
      attention_layer(mix, hn,
                      tensor_of(hbm, W_QP[layer]), tensor_of(hbm, W_KP[layer]),
                      tensor_of(hbm, W_VP[layer]), tensor_of(hbm, W_OP[layer]),
                      f32_of(hbm, W_QN[layer]), f32_of(hbm, W_KN[layer]),
                      st.pos, st.kc[fi].data(), st.vc[fi].data(), st.pos);
      ++fi;
    }
    for (int i = 0; i < HIDDEN; ++i) h[i] = r[i] + mix[i];

    for (int i = 0; i < HIDDEN; ++i) r[i] = h[i];
    rmsnorm(hn, h, f32_of(hbm, W_POST_LN[layer]), HIDDEN, EPS);
    mlp_layer(mix, hn, tensor_of(hbm, W_GATE[layer]),
              tensor_of(hbm, W_UP[layer]), tensor_of(hbm, W_DOWN[layer]));
    for (int i = 0; i < HIDDEN; ++i) h[i] = r[i] + mix[i];
  }

  rmsnorm(h, h, f32_of(hbm, W_FINAL), HIDDEN, EPS);

  int best = 0;
  float bestv = -1e30f;
LMHEAD:
  for (int vtok = 0; vtok < VOCAB; ++vtok) {
    const float l = row_dot_i8(emb.w_i8, emb.scale, h, vtok, HIDDEN);
    if (logits) logits[vtok] = l;
    if (l > bestv) { bestv = l; best = vtok; }
  }

  ++st.pos;
  return best;
}

}  // namespace qwen
