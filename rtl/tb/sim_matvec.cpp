// Verilator testbench for the full int8 matvec primitive (GEMV + dequant).
// Streams a real k_proj case, dequantises with the host-computed fixed-point
// multiplier, and checks the Q14 output against the model's fp32 result.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vmatvec_i8.h"
#include "verilated.h"

static Vmatvec_i8* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vmatvec_i8;
  const int IN = 1024, OUT = 512, LANES = 8, WPR = IN / LANES;
  const int SHIFT = 16, OUTQ = 14;

  std::ifstream f("artifacts/tv_gemv.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_gemv_tv.py first\n"); return 2; }
  std::vector<int8_t> x(IN), w((size_t)OUT * IN);
  std::vector<int32_t> acc_ref(OUT);
  float sx;
  std::vector<float> sw(OUT), y_ref(OUT);
  f.read((char*)x.data(), IN);
  f.read((char*)w.data(), (size_t)OUT * IN);
  f.read((char*)acc_ref.data(), 4 * OUT);
  f.read((char*)&sx, 4);
  f.read((char*)sw.data(), 4 * OUT);
  f.read((char*)y_ref.data(), 4 * OUT);

  // host-side fixed-point multipliers, one per row
  std::vector<int32_t> mult(OUT);
  for (int o = 0; o < OUT; o++) {
    double c = (double)sx * (double)sw[o];
    mult[o] = (int32_t)std::llround(c * std::pow(2.0, OUTQ + SHIFT));
  }

  dut->rst = 1; dut->load_en = 0; dut->w_en = 0; dut->mult = 0;
  tick(); tick(); dut->rst = 0;
  for (int i = 0; i < IN; i++) { dut->load_en = 1; dut->x_byte = x[i]; tick(); }
  dut->load_en = 0;

  std::vector<int16_t> got;
  got.reserve(OUT);
  for (int o = 0; o < OUT; o++) {
    dut->mult = mult[o];
    for (int c = 0; c < WPR; c++) {
      uint64_t word = 0;
      for (int j = 0; j < LANES; j++)
        word |= (uint64_t)(uint8_t)w[(size_t)o * IN + c * LANES + j] << (j * 8);
      dut->w_en = 1; dut->w_word = word;
      tick();
      if (dut->y_vld) got.push_back((int16_t)dut->y_q);
    }
  }

  // compare Q14 output against round(y_fp32 * 2^14)
  int n = (int)got.size(), bad = 0, worst = 0;
  for (int o = 0; o < OUT && o < n; o++) {
    int ref = (int)std::lround((double)y_ref[o] * (1 << OUTQ));
    int d = std::abs((int)got[o] - ref);
    if (d > worst) worst = d;
    if (d > 4) { if (bad < 5) std::printf("  row %d: got %d exp %d (d=%d)\n", o, got[o], ref, d); bad++; }
  }
  std::printf("collected %d of %d rows, worst |diff| = %d LSB (Q%d)\n", n, OUT, worst, OUTQ);
  bool ok = (bad == 0 && n == OUT);
  std::printf("%s\n", ok ? "PASS  matvec_i8 == model fp32 (within fixed-point rounding)" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
