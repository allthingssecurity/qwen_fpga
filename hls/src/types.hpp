// Qwen3.5-0.8B text decode -- shapes, constants, and the HBM weight-blob view.
//
// Every dimension here is pinned to the checkpoint's config.json (text_config)
// and matches golden/model.py, which is bit-verified against HF transformers.
// Nothing is a round number for convenience; if it disagrees with the config,
// the config wins.
//
// Synthesis note: the kernels in the other headers are written to synthesise
// under Vitis (float datapaths, #pragma HLS directives). The top-level
// orchestration in decoder.hpp addresses weights in HBM by byte offset -- the
// same blob pack_weights.py wrote -- so the decode is a single sweep through
// HBM, exactly as the bandwidth harness modelled.

#pragma once
#include <cstdint>

namespace qwen {

// ---- global model dims
constexpr int HIDDEN = 1024;
constexpr int N_LAYERS = 24;
constexpr int INTER = 3584;   // MLP intermediate
constexpr int VOCAB = 248320;
constexpr float EPS = 1e-6f;

// ---- Gated DeltaNet (linear-attention layers)
constexpr int LH = 16;        // linear_num_value_heads (== key heads)
constexpr int LK = 128;       // linear_key_head_dim
constexpr int LV = 128;       // linear_value_head_dim
constexpr int CONV_DIM = 6144;   // key*2 + value = 2048*2 + 2048
constexpr int CONV_K = 4;        // linear_conv_kernel_dim
constexpr int Z_DIM = 2048;      // in_proj_z out
constexpr int QKV_DIM = 6144;    // in_proj_qkv out
constexpr int VALUE_DIM = 2048;  // LH*LV, == out_proj in

// ---- full attention layers
constexpr int NH = 8;         // num_attention_heads
constexpr int NKV = 2;        // num_key_value_heads
constexpr int HD = 256;       // head_dim
constexpr int Q_PROJ = NH * HD * 2;   // 4096: query + output-gate interleaved per head
constexpr int KV_PROJ = NKV * HD;     // 512
constexpr int ATTN_O_IN = NH * HD;    // 2048: o_proj input
constexpr int ROPE_DIM = 64;          // int(head_dim * partial_rotary_factor 0.25)
constexpr float ROPE_THETA = 10000000.0f;

// ---- layer schedule: [linear x3, full] x6
inline bool is_linear(int layer) { return (layer % 4) != 3; }

// ---- a weight tensor's location in the HBM blob (filled from the manifest).
// int8 tensors carry a per-output-row fp32 scale; fp32 tensors don't.
struct Tensor {
  const int8_t* w_i8 = nullptr;    // int8 payload (row-major [out,in]), or null
  const float* scale = nullptr;    // per-row fp32 scale for int8, else null
  const float* w_f32 = nullptr;    // fp32 payload (norms, A_log, dt_bias, conv, in_proj_a/b)
  int out = 0, in = 0;
};

}  // namespace qwen
