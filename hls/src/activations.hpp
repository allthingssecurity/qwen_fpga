// Scalar nonlinearities, matched to golden/model.py (which matches torch).
#pragma once
#include <cmath>

namespace qwen {

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// softplus with the same large-x guard torch/golden use: log1p(exp(x)), but
// pass x through unchanged above 20 to avoid overflow.
inline float softplusf(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

}  // namespace qwen
