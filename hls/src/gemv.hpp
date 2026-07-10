// The workhorse: weight-stationary streaming GEMV. y[out] = W[out,in] . x[in].
//
// This is where ~99% of the HBM bandwidth and the compute go. For batch-1
// decode it's a matrix-VECTOR product: each weight is read once, multiplied
// once, discarded -- arithmetic intensity ~2 op/byte, the bandwidth-bound
// regime the whole design is built around.
//
// int8 path: weights stream from HBM as int8; the fp32 activation x stays
// resident on-chip. Accumulate int8*fp32 in fp32 per output row, then apply the
// single per-row scale. Factoring the scale out of the inner sum is what makes
// this cheap in hardware (one multiply per output, not per MAC) and is exact in
// real arithmetic; only fp summation ORDER differs from the numpy golden, which
// dequantises first -- a ~1e-4 reassociation gap, not an error.
//
// Dataflow shape for synthesis: PIPELINE the output-row loop, UNROLL a chunk of
// the inner reduction so a block of int8 weights is consumed per cycle and the
// HBM read stream never stalls. The row is the natural unit to map onto one HBM
// burst.

#pragma once
#include "types.hpp"

namespace qwen {

constexpr int GEMV_UNROLL = 16;   // inner-reduction lanes; tune for DSP/BRAM balance

// int8 weights [out,in] row-major, per-row scale[out]; x[in] fp32 -> y[out] fp32
inline void gemv_i8(float* y, const int8_t* w, const float* scale,
                    const float* x, int out, int in) {
GEMV_ROW:
  for (int o = 0; o < out; ++o) {
#pragma HLS PIPELINE II = 1
    const int8_t* row = w + (long)o * in;
    float acc = 0.0f;
  GEMV_DOT:
    for (int i = 0; i < in; ++i) {
#pragma HLS UNROLL factor = GEMV_UNROLL
      acc += float(row[i]) * x[i];
    }
    y[o] = acc * scale[o];
  }
}

// fp32 weights [out,in] row-major; x[in] -> y[out]. For the handful of small
// tensors kept in fp32 (in_proj_a/b). Bit-exact accumulation order vs numpy.
inline void gemv_f32(float* y, const float* w, const float* x, int out, int in) {
GEMVF_ROW:
  for (int o = 0; o < out; ++o) {
#pragma HLS PIPELINE II = 1
    const float* row = w + (long)o * in;
    float acc = 0.0f;
  GEMVF_DOT:
    for (int i = 0; i < in; ++i) {
#pragma HLS UNROLL factor = GEMV_UNROLL
      acc += row[i] * x[i];
    }
    y[o] = acc;
  }
}

// One int8 embedding/lm_head row dotted with h -> a single logit. Used to stream
// the 248320-row tied head without materialising all logits at once.
inline float row_dot_i8(const int8_t* w, const float* scale, const float* h,
                        int row, int in) {
  const int8_t* r = w + (long)row * in;
  float acc = 0.0f;
DOT:
  for (int i = 0; i < in; ++i) {
#pragma HLS UNROLL factor = GEMV_UNROLL
    acc += float(r[i]) * h[i];
  }
  return acc * scale[row];
}

// Embedding lookup: dequantise one int8 row into h[HIDDEN].
inline void embed_row(float* h, const int8_t* w, const float* scale, int tok, int in) {
  const int8_t* r = w + (long)tok * in;
  const float s = scale[tok];
EMB:
  for (int i = 0; i < in; ++i) {
#pragma HLS UNROLL factor = GEMV_UNROLL
    h[i] = float(r[i]) * s;
  }
}

}  // namespace qwen
