// SwiGLU MLP: down( silu(gate(x)) * up(x) ). Three int8 GEMVs and an
// elementwise product. Same in every layer. Mirrors golden/model.py mlp.
#pragma once
#include "activations.hpp"
#include "gemv.hpp"
#include "types.hpp"

namespace qwen {

// out[HIDDEN]; x[HIDDEN] post-attn-normed.
inline void mlp_layer(float* out, const float* x,
                      const Tensor& gate_w, const Tensor& up_w, const Tensor& down_w) {
  static float g[INTER], u[INTER];
  gemv_i8(g, gate_w.w_i8, gate_w.scale, x, INTER, HIDDEN);
  gemv_i8(u, up_w.w_i8, up_w.scale, x, INTER, HIDDEN);
MLP_ACT:
  for (int i = 0; i < INTER; ++i) g[i] = silu(g[i]) * u[i];
  gemv_i8(out, down_w.w_i8, down_w.scale, g, HIDDEN, INTER);
}

}  // namespace qwen
