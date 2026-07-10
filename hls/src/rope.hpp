// Partial rotary position embedding for the full-attention layers.
//
// Text-only collapses mRoPE to plain partial RoPE: with no image tokens the
// T/H/W position ids are identical, so apply_interleaved_mrope is a no-op. Only
// the first ROPE_DIM (64) of each 256-dim head is rotated; the rest passes
// through. See golden/model.py rope_cos_sin / apply_rope.

#pragma once
#include <cmath>

#include "types.hpp"

namespace qwen {

// cos/sin for a given position. Each has ROPE_DIM entries; the first ROPE_DIM/2
// mirror into the second half (emb = concat(f, f)).
inline void rope_cos_sin(int pos, float cos_o[ROPE_DIM], float sin_o[ROPE_DIM]) {
  constexpr int HALF = ROPE_DIM / 2;   // 32
ROPE_FREQ:
  for (int j = 0; j < HALF; ++j) {
    const float inv = 1.0f / std::pow(ROPE_THETA, float(2 * j) / ROPE_DIM);
    const float f = pos * inv;
    const float c = std::cos(f), s = std::sin(f);
    cos_o[j] = c; cos_o[j + HALF] = c;
    sin_o[j] = s; sin_o[j + HALF] = s;
  }
}

// Rotate the first ROPE_DIM dims of one head in place; leave [ROPE_DIM..HD).
inline void apply_rope(float* x, const float cos_[ROPE_DIM], const float sin_[ROPE_DIM]) {
  constexpr int HALF = ROPE_DIM / 2;
  float rot[ROPE_DIM];
ROPE_CP:
  for (int i = 0; i < ROPE_DIM; ++i) rot[i] = x[i];
ROPE_APPLY:
  for (int i = 0; i < ROPE_DIM; ++i) {
    // rotate_half: [-x2, x1] with x1=rot[:HALF], x2=rot[HALF:]
    const float half = (i < HALF) ? -rot[i + HALF] : rot[i - HALF];
    x[i] = rot[i] * cos_[i] + half * sin_[i];
  }
}

}  // namespace qwen
