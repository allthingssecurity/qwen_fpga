// End-to-end decode of one token: embed -> 24 layers -> final norm -> tied head.
//
// The per-op kernels (gemv/norm/rope/conv/deltanet/attention/mlp) are the
// synthesisable fabric. This file is the SEQUENCER: it walks the layer schedule
// and hands each kernel its weight pointers into the HBM blob. In a real Vitis
// build this loop is either a small control FSM or host-side orchestration; the
// arithmetic that becomes gates lives entirely in the kernels it calls.
//
// Weight addressing is by byte offset into the single packed blob
// (artifacts/qwen35_int8.bin), in the exact order pack_weights.py laid them out
// -- so one token is one forward sweep through HBM, matching the bandwidth
// harness.

#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "attention.hpp"
#include "deltanet.hpp"
#include "gemv.hpp"
#include "mlp.hpp"
#include "norm.hpp"
#include "state.hpp"
#include "types.hpp"

namespace qwen {

// ---- host-side view of the packed weight blob (built from the manifest index)
struct Entry {
  uint64_t off = 0, scale_off = 0;
  int dtype = 0;   // 0 fp32, 1 int8
  int out = 0, in = 0;
};

struct WeightBlob {
  const uint8_t* base = nullptr;
  std::unordered_map<std::string, Entry> idx;

  Tensor get(const std::string& n) const {
    const Entry& e = idx.at(n);
    Tensor t;
    t.out = e.out;
    t.in = e.in;
    if (e.dtype == 1) {
      t.w_i8 = reinterpret_cast<const int8_t*>(base + e.off);
      t.scale = reinterpret_cast<const float*>(base + e.scale_off);
    } else {
      t.w_f32 = reinterpret_cast<const float*>(base + e.off);
    }
    return t;
  }
  const float* f32(const std::string& n) const {
    return reinterpret_cast<const float*>(base + idx.at(n).off);
  }
};

// DecodeState now lives in state.hpp (shared with the synthesisable top).

// Decode one token. If logits != null, writes all VOCAB logits. Returns argmax.
inline int decode_token(const WeightBlob& W, DecodeState& st, int tok, float* logits) {
  static float h[HIDDEN], hn[HIDDEN], mix[HIDDEN], r[HIDDEN];

  const Tensor emb = W.get("embed_tokens.weight");
  embed_row(h, emb.w_i8, emb.scale, tok, HIDDEN);

  int li = 0, fi = 0;
  for (int layer = 0; layer < N_LAYERS; ++layer) {
    const std::string p = "layers." + std::to_string(layer);

    // token mixer, pre-normed with residual
    for (int i = 0; i < HIDDEN; ++i) r[i] = h[i];
    rmsnorm(hn, h, W.f32(p + ".input_layernorm.weight"), HIDDEN, EPS);

    if (is_linear(layer)) {
      const std::string a = p + ".linear_attn";
      deltanet_layer(mix, hn,
                     W.get(a + ".in_proj_qkv.weight"), W.get(a + ".in_proj_z.weight"),
                     W.get(a + ".in_proj_b.weight"), W.get(a + ".in_proj_a.weight"),
                     W.f32(a + ".conv1d.weight"), W.f32(a + ".A_log"),
                     W.f32(a + ".dt_bias"), W.f32(a + ".norm.weight"),
                     W.get(a + ".out_proj.weight"),
                     st.rec[li], st.conv[li]);
      ++li;
    } else {
      const std::string a = p + ".self_attn";
      st.kc[fi].resize((long)(st.pos + 1) * KV_PROJ);
      st.vc[fi].resize((long)(st.pos + 1) * KV_PROJ);
      attention_layer(mix, hn,
                      W.get(a + ".q_proj.weight"), W.get(a + ".k_proj.weight"),
                      W.get(a + ".v_proj.weight"), W.get(a + ".o_proj.weight"),
                      W.f32(a + ".q_norm.weight"), W.f32(a + ".k_norm.weight"),
                      st.pos, st.kc[fi].data(), st.vc[fi].data(), st.pos);
      ++fi;
    }
    for (int i = 0; i < HIDDEN; ++i) h[i] = r[i] + mix[i];

    // MLP, pre-normed with residual
    for (int i = 0; i < HIDDEN; ++i) r[i] = h[i];
    rmsnorm(hn, h, W.f32(p + ".post_attention_layernorm.weight"), HIDDEN, EPS);
    mlp_layer(mix, hn, W.get(p + ".mlp.gate_proj.weight"),
              W.get(p + ".mlp.up_proj.weight"), W.get(p + ".mlp.down_proj.weight"));
    for (int i = 0; i < HIDDEN; ++i) h[i] = r[i] + mix[i];
  }

  rmsnorm(h, h, W.f32("norm.weight"), HIDDEN, EPS);

  // tied lm_head: stream the 248320 int8 rows, keep argmax (+ optional logits)
  int best = 0;
  float bestv = -1e30f;
  for (int vtok = 0; vtok < VOCAB; ++vtok) {
    const float l = row_dot_i8(emb.w_i8, emb.scale, h, vtok, HIDDEN);
    if (logits) logits[vtok] = l;
    if (l > bestv) { bestv = l; best = vtok; }
  }

  ++st.pos;
  return best;
}

}  // namespace qwen
