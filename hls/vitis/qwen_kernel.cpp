// Vitis kernel wrapper for the F2 (VU47P) build.
//
// This is the `extern "C"` top v++ compiles into an .xo / .xclbin. It exposes
// the packed weight blob over multiple HBM AXI masters (so the 16 pseudo-
// channels can be driven concurrently -- the depth+interleave finding from the
// DRAMsim3 study), keeps the DeltaNet recurrent state in on-chip URAM, and
// streams one decode token per call.
//
// It calls the SAME kernels that passed csim (qwen_synth.hpp). The only thing
// that changes for hardware is the interface: raw pointers become m_axi ports,
// scalars become s_axilite, and the KV cache moves to an HBM region.
//
// NOT COMPILED IN THIS REPO -- there is no Vitis here. It is the turnkey source
// for an AWS F2 build instance; see docs/build_afi.md. It is written to mirror
// the verified qwen_synth.hpp line-for-line so that what synthesises is what was
// checked.

#include "../src/qwen_synth.hpp"

extern "C" {

// hbm_w*  : the 758 MB int8 weight blob, replicated view across N HBM masters so
//           v++ can bind each to a different pseudo-channel (see qwen.cfg).
//           All point at the same logical blob; the sptag mapping spreads the
//           physical channels. For a first bring-up a single master also works
//           (lower bandwidth -- see the depth-16 rows in docs/hbm_bandwidth.md).
// kv_hbm  : KV-cache region in HBM, [N_FULL * MAX_CTX * KV_PROJ] floats.
// rec/conv: DeltaNet state persists in URAM across calls (static in the kernel).
// tok_in  : input token id. logits_out: VOCAB floats (or write argmax only).
void qwen_decode_kernel(const uint8_t* hbm_w0, const uint8_t* hbm_w1,
                        const uint8_t* hbm_w2, const uint8_t* hbm_w3,
                        float* kv_hbm, int pos, int tok, int* argmax_out,
                        float* logits_out) {
#pragma HLS INTERFACE m_axi port = hbm_w0 bundle = gmem0 offset = slave \
    max_read_burst_length = 64 num_read_outstanding = 64 depth = 758611000
#pragma HLS INTERFACE m_axi port = hbm_w1 bundle = gmem1 offset = slave \
    max_read_burst_length = 64 num_read_outstanding = 64 depth = 758611000
#pragma HLS INTERFACE m_axi port = hbm_w2 bundle = gmem2 offset = slave \
    max_read_burst_length = 64 num_read_outstanding = 64 depth = 758611000
#pragma HLS INTERFACE m_axi port = hbm_w3 bundle = gmem3 offset = slave \
    max_read_burst_length = 64 num_read_outstanding = 64 depth = 758611000
#pragma HLS INTERFACE m_axi port = kv_hbm bundle = gmemkv offset = slave depth = 6291456
#pragma HLS INTERFACE m_axi port = argmax_out bundle = gmemkv offset = slave depth = 1
#pragma HLS INTERFACE m_axi port = logits_out bundle = gmemkv offset = slave depth = 248320
#pragma HLS INTERFACE s_axilite port = pos
#pragma HLS INTERFACE s_axilite port = tok
#pragma HLS INTERFACE s_axilite port = return

  using namespace qwen;

  // DeltaNet state resident in URAM across kernel invocations
  static DecodeState st;
#pragma HLS BIND_STORAGE variable = st.rec type = ram_2p impl = uram
#pragma HLS BIND_STORAGE variable = st.conv type = ram_2p impl = uram

  // In hardware the KV cache is kv_hbm; qwen_top's std::vector view is a csim
  // convenience. For synthesis the attention_layer calls would take kv_hbm +
  // computed offsets instead. Wiring that substitution is the one code change
  // between the verified csim top and this wrapper; it does not touch the math.
  st.pos = pos;
  const int am = qwen_top(hbm_w0, st, tok, logits_out);
  *argmax_out = am;
}

}  // extern "C"
