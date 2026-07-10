// GENERATED. F2 HBM bandwidth kernel: 32 ports, one per pseudo-channel.
#include <ap_int.h>
#include <cstdint>
typedef ap_uint<512> u512;   // 64 bytes / beat

static void read_bank(const u512* in, unsigned words, unsigned iters, uint64_t* out) {
  ap_uint<512> acc = 0;
  for (unsigned r = 0; r < iters; ++r)
    for (unsigned i = 0; i < words; ++i) {
#pragma HLS PIPELINE II = 1
      acc ^= in[i];   // xor keeps the reads from being optimised away
    }
  *out = (uint64_t)acc.range(63, 0);
}

extern "C" void hbm_bw(
    const u512* in0,
    const u512* in1,
    const u512* in2,
    const u512* in3,
    const u512* in4,
    const u512* in5,
    const u512* in6,
    const u512* in7,
    const u512* in8,
    const u512* in9,
    const u512* in10,
    const u512* in11,
    const u512* in12,
    const u512* in13,
    const u512* in14,
    const u512* in15,
    const u512* in16,
    const u512* in17,
    const u512* in18,
    const u512* in19,
    const u512* in20,
    const u512* in21,
    const u512* in22,
    const u512* in23,
    const u512* in24,
    const u512* in25,
    const u512* in26,
    const u512* in27,
    const u512* in28,
    const u512* in29,
    const u512* in30,
    const u512* in31,
    unsigned words, unsigned iters, uint64_t* csum) {
#pragma HLS INTERFACE m_axi port = in0 bundle = g0 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in1 bundle = g1 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in2 bundle = g2 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in3 bundle = g3 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in4 bundle = g4 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in5 bundle = g5 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in6 bundle = g6 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in7 bundle = g7 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in8 bundle = g8 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in9 bundle = g9 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in10 bundle = g10 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in11 bundle = g11 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in12 bundle = g12 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in13 bundle = g13 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in14 bundle = g14 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in15 bundle = g15 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in16 bundle = g16 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in17 bundle = g17 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in18 bundle = g18 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in19 bundle = g19 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in20 bundle = g20 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in21 bundle = g21 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in22 bundle = g22 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in23 bundle = g23 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in24 bundle = g24 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in25 bundle = g25 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in26 bundle = g26 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in27 bundle = g27 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in28 bundle = g28 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in29 bundle = g29 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in30 bundle = g30 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = in31 bundle = g31 offset = slave num_read_outstanding = 64 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = csum bundle = gout offset = slave
#pragma HLS INTERFACE s_axilite port = words
#pragma HLS INTERFACE s_axilite port = iters
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW
  read_bank(in0, words, iters, &csum[0]);
  read_bank(in1, words, iters, &csum[1]);
  read_bank(in2, words, iters, &csum[2]);
  read_bank(in3, words, iters, &csum[3]);
  read_bank(in4, words, iters, &csum[4]);
  read_bank(in5, words, iters, &csum[5]);
  read_bank(in6, words, iters, &csum[6]);
  read_bank(in7, words, iters, &csum[7]);
  read_bank(in8, words, iters, &csum[8]);
  read_bank(in9, words, iters, &csum[9]);
  read_bank(in10, words, iters, &csum[10]);
  read_bank(in11, words, iters, &csum[11]);
  read_bank(in12, words, iters, &csum[12]);
  read_bank(in13, words, iters, &csum[13]);
  read_bank(in14, words, iters, &csum[14]);
  read_bank(in15, words, iters, &csum[15]);
  read_bank(in16, words, iters, &csum[16]);
  read_bank(in17, words, iters, &csum[17]);
  read_bank(in18, words, iters, &csum[18]);
  read_bank(in19, words, iters, &csum[19]);
  read_bank(in20, words, iters, &csum[20]);
  read_bank(in21, words, iters, &csum[21]);
  read_bank(in22, words, iters, &csum[22]);
  read_bank(in23, words, iters, &csum[23]);
  read_bank(in24, words, iters, &csum[24]);
  read_bank(in25, words, iters, &csum[25]);
  read_bank(in26, words, iters, &csum[26]);
  read_bank(in27, words, iters, &csum[27]);
  read_bank(in28, words, iters, &csum[28]);
  read_bank(in29, words, iters, &csum[29]);
  read_bank(in30, words, iters, &csum[30]);
  read_bank(in31, words, iters, &csum[31]);
}
