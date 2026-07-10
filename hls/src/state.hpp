// Persistent decode state, shared by the csim orchestrator (decoder.hpp) and
// the synthesisable top (qwen_synth.hpp).
//
// In hardware: rec/conv live in on-chip URAM (~20 MB, never leave the die); the
// KV cache lives in HBM. Here the KV cache is std::vector for csim convenience
// -- the Vitis kernel wrapper replaces those with an HBM region (see
// hls/vitis/). rec/conv map 1:1 to URAM.
#pragma once
#include <vector>

#include "types.hpp"

namespace qwen {

constexpr int N_LINEAR = 18;
constexpr int N_FULL = 6;

struct DecodeState {
  float rec[N_LINEAR][LH][LK][LV];              // DeltaNet recurrent state (18 MB) -> URAM
  float conv[N_LINEAR][CONV_DIM][CONV_K];       // conv history -> URAM
  std::vector<float> kc[N_FULL], vc[N_FULL];    // KV cache -> HBM in hardware
  int pos = 0;

  void reset() {
    for (int l = 0; l < N_LINEAR; ++l) {
      for (int h = 0; h < LH; ++h)
        for (int a = 0; a < LK; ++a)
          for (int b = 0; b < LV; ++b) rec[l][h][a][b] = 0.0f;
      for (int c = 0; c < CONV_DIM; ++c)
        for (int k = 0; k < CONV_K; ++k) conv[l][c][k] = 0.0f;
    }
    for (int f = 0; f < N_FULL; ++f) { kc[f].clear(); vc[f].clear(); }
    pos = 0;
  }
};

}  // namespace qwen
