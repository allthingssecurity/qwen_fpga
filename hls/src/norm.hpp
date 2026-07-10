// The two RMSNorms. Getting these confused produces fluent garbage, not a
// crash -- see golden/model.py, where mixing them was the first bug caught.
#pragma once
#include <cmath>

#include "activations.hpp"

namespace qwen {

// Qwen3_5RMSNorm: normalise over `n`, scale by (1 + w). The checkpoint stores
// gamma ZERO-CENTERED, so the gain is (1 + weight), not weight. Used for the
// input/post layernorms, q_norm, k_norm, and the final norm.
inline void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
  float ss = 0.0f;
RMS_SUM:
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  const float inv = 1.0f / std::sqrt(ss / n + eps);
RMS_SCALE:
  for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (1.0f + w[i]);
}

// Qwen3_5RMSNormGated (DeltaNet only): normalise over `n`, scale by w directly
// (NO 1+), then gate with silu(z). This weight is ones-initialised -- the one
// F32 tensor in the bf16 checkpoint.
inline void rmsnorm_gated(float* y, const float* x, const float* w,
                          const float* z, int n, float eps) {
  float ss = 0.0f;
RMSG_SUM:
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  const float inv = 1.0f / std::sqrt(ss / n + eps);
RMSG_SCALE:
  for (int i = 0; i < n; ++i) y[i] = x[i] * inv * w[i] * silu(z[i]);
}

}  // namespace qwen
