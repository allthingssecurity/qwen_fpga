// Verilator testbench for fixed-point softmax, vs numpy softmax.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vsoftmax.h"
#include "verilated.h"

static Vsoftmax* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }
static int32_t q10(double x) { return (int32_t)llround(x * 1024.0); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vsoftmax;

  std::ifstream f("artifacts/tv_softmax.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_softmax_tv.py first\n"); return 2; }
  int32_t hdr[2]; f.read((char*)hdr, 8);
  int T = hdr[0], LUTN = hdr[1];
  std::vector<float> lut(LUTN), scores(T), w(T);
  f.read((char*)lut.data(), 4 * LUTN);
  f.read((char*)scores.data(), 4 * T);
  f.read((char*)w.data(), 4 * T);

  dut->rst = 1; dut->start = 0; dut->we_lut = 0; dut->we_s = 0;
  tick(); tick(); dut->rst = 0;
  for (int i = 0; i < LUTN; i++) {
    dut->we_lut = 1; dut->addr_lut = i;
    dut->din_lut = (uint32_t)llround((double)lut[i] * 65536.0);  // Q16
    tick();
  }
  dut->we_lut = 0;
  for (int i = 0; i < T; i++) { dut->we_s = 1; dut->addr_s = i; dut->din_s = q10(scores[i]); tick(); }
  dut->we_s = 0;

  dut->n_scores = T;
  dut->start = 1; tick(); dut->start = 0;
  int guard = 0;
  while (!dut->done && guard++ < 5000) tick();
  if (!dut->done) { std::printf("FAIL: never finished\n"); return 1; }

  double worst = 0, sum = 0;
  for (int i = 0; i < T; i++) {
    dut->raddr_w = i; dut->eval();
    double got = (double)(uint32_t)dut->rdata_w / 65536.0;
    sum += got;
    worst = std::max(worst, std::fabs(got - (double)w[i]));
  }
  std::printf("worst |diff| %.3e  (weights sum to %.4f, should be 1.0)\n", worst, sum);
  bool ok = (worst < 2e-3) && std::fabs(sum - 1.0) < 5e-3;
  std::printf("%s\n", ok ? "PASS  softmax == numpy (fixed point)" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
