// Representative int8 GEMV for a Vitis HLS resource/timing report: the MLP
// gate/up projection shape [3584 x 1024], fixed bounds so trip counts (and thus
// II/latency) are concrete. This is the workhorse kernel -- if it closes timing
// at II=1 and fits, the bandwidth story holds.
#include <cstdint>

void synth_gemv_mlp(float* y, const int8_t* w, const float* scale, const float* x) {
#pragma HLS INTERFACE m_axi port = w bundle = gmem0 depth = 3670016
#pragma HLS INTERFACE m_axi port = scale bundle = gmem1 depth = 3584
#pragma HLS INTERFACE m_axi port = x bundle = gmem1 depth = 1024
#pragma HLS INTERFACE m_axi port = y bundle = gmem1 depth = 3584
#pragma HLS INTERFACE s_axilite port = return
ROW:
  for (int o = 0; o < 3584; ++o) {
#pragma HLS PIPELINE II = 1
    const int8_t* row = w + (long)o * 1024;
    float acc = 0.0f;
  DOT:
    for (int i = 0; i < 1024; ++i) {
#pragma HLS UNROLL factor = 16
      acc += float(row[i]) * x[i];
    }
    y[o] = acc * scale[o];
  }
}
