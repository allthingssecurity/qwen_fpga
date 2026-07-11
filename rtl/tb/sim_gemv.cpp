// Verilator testbench: drive gemv_i8 with a real int8 GEMV case from the model
// and check the int32 accumulators exactly against numpy.
//
//   python3 scripts/export_gemv_tv.py
//   verilator --cc --exe --build -Irtl rtl/gemv_i8.sv rtl/tb/sim_gemv.cpp \
//       --top-module gemv_i8 -o sim_gemv
//   ./obj_dir/sim_gemv         (run from repo root so the tv path resolves)

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vgemv_i8.h"
#include "verilated.h"

static Vgemv_i8* dut;

static void tick() {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vgemv_i8;

  const int IN = 1024, OUT = 512, LANES = 8, WPR = IN / LANES;

  std::ifstream f("artifacts/tv_gemv.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_gemv_tv.py first\n"); return 2; }
  std::vector<int8_t> x(IN);
  std::vector<int8_t> w((size_t)OUT * IN);
  std::vector<int32_t> acc_ref(OUT);
  f.read((char*)x.data(), IN);
  f.read((char*)w.data(), (size_t)OUT * IN);
  f.read((char*)acc_ref.data(), 4 * OUT);

  dut->rst = 1; dut->load_en = 0; dut->w_en = 0;
  tick(); tick();
  dut->rst = 0;

  for (int i = 0; i < IN; i++) { dut->load_en = 1; dut->x_byte = x[i]; tick(); }
  dut->load_en = 0;

  std::vector<int32_t> got;
  got.reserve(OUT);
  for (int o = 0; o < OUT; o++) {
    for (int c = 0; c < WPR; c++) {
      uint64_t word = 0;
      for (int j = 0; j < LANES; j++) {
        uint8_t b = (uint8_t)w[(size_t)o * IN + c * LANES + j];
        word |= (uint64_t)b << (j * 8);
      }
      dut->w_en = 1; dut->w_word = word;
      tick();
      if (dut->acc_vld) got.push_back((int32_t)dut->acc_o);
    }
  }
  dut->w_en = 0;
  for (int k = 0; k < 3; k++) { tick(); if (dut->acc_vld) got.push_back((int32_t)dut->acc_o); }

  int n = (int)got.size(), mism = 0;
  for (int o = 0; o < OUT && o < n; o++) {
    if (got[o] != acc_ref[o]) {
      if (mism < 5) std::printf("  mismatch row %d: got %d exp %d\n", o, got[o], acc_ref[o]);
      mism++;
    }
  }
  std::printf("collected %d of %d output rows\n", n, OUT);
  bool ok = (mism == 0 && n == OUT);
  std::printf("%s\n", ok ? "PASS  gemv_i8 == golden int8 (exact)" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
