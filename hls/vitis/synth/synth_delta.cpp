// The Gated DeltaNet recurrent head, for a Vitis HLS report. Fixed 128x128
// state, the two-sweep delta rule -- the design's most unusual datapath.
#include "../../src/deltanet.hpp"

void synth_delta_head(const float qry[128], const float key[128], const float val[128],
                      float g, float beta, float S[128][128], float out[128]) {
#pragma HLS INTERFACE s_axilite port = return
  qwen::delta_rule_head(qry, key, val, g, beta, S, out);
}
