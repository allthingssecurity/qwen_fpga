// Depthwise causal conv1d, kernel 4, then SiLU -- the short-convolution front
// of each Gated DeltaNet layer.
//
// Per channel we keep CONV_K taps of history. On a decode step we append the
// new sample, shift, and convolve. torch keeps 4 taps but the causal output
// only reads taps [1..4] (tap 0 is always discarded); we keep 4 to stay
// tensor-identical with torch's conv_state. See golden/model.py gated_deltanet.

#pragma once
#include "activations.hpp"
#include "types.hpp"

namespace qwen {

// state: [CONV_DIM][CONV_K] history (persisted across tokens, on-chip in HW).
// in_ch[CONV_DIM] this step's pre-conv qkv. out[CONV_DIM] silu(conv).
inline void conv1d_step(float* out, const float* in_ch, float state[CONV_DIM][CONV_K],
                        const float* weight /* [CONV_DIM][CONV_K] */) {
CONV_CH:
  for (int c = 0; c < CONV_DIM; ++c) {
#pragma HLS PIPELINE II = 1
    // window = [state[c][1..3], new] ; then history becomes last 4 = same window
    float win[CONV_K];
    win[0] = state[c][1];
    win[1] = state[c][2];
    win[2] = state[c][3];
    win[3] = in_ch[c];
    float acc = 0.0f;
    const float* w = weight + c * CONV_K;
  CONV_TAP:
    for (int k = 0; k < CONV_K; ++k) {
#pragma HLS UNROLL
      acc += win[k] * w[k];
      state[c][k] = win[k];   // shift history forward
    }
    out[c] = silu(acc);
  }
}

}  // namespace qwen
