// Verilator testbench for partial RoPE, vs golden apply_rope.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vrope.h"
#include "verilated.h"

static Vrope* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }
static int32_t q12(double x) { return (int32_t)llround(x * 4096.0); }
static int32_t q14(double x) { return (int32_t)llround(x * 16384.0); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vrope;

  std::ifstream f("artifacts/tv_rope.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_rope_tv.py first\n"); return 2; }
  int32_t hdr[2]; f.read((char*)hdr, 8);
  int HD = hdr[0], RD = hdr[1];
  std::vector<float> q(HD), cos(RD), sin(RD), out(HD);
  f.read((char*)q.data(), 4 * HD);
  f.read((char*)cos.data(), 4 * RD);
  f.read((char*)sin.data(), 4 * RD);
  f.read((char*)out.data(), 4 * HD);

  dut->rst = 1; dut->start = 0; dut->we_q = 0; dut->we_cs = 0;
  tick(); tick(); dut->rst = 0;
  for (int i = 0; i < HD; i++) { dut->we_q = 1; dut->addr_q = i; dut->din_q = q12(q[i]); tick(); } dut->we_q = 0;
  for (int i = 0; i < RD; i++) { dut->we_cs = 1; dut->addr_cs = i;      dut->din_cs = q14(cos[i]); tick(); }
  for (int i = 0; i < RD; i++) { dut->we_cs = 1; dut->addr_cs = RD + i; dut->din_cs = q14(sin[i]); tick(); }
  dut->we_cs = 0;

  dut->start = 1; tick(); dut->start = 0;
  int guard = 0;
  while (!dut->done && guard++ < 2000) tick();
  if (!dut->done) { std::printf("FAIL: never finished\n"); return 1; }

  double worst = 0, refmax = 0;
  for (int i = 0; i < HD; i++) {
    dut->raddr_o = i; dut->eval();
    double got = (double)(int32_t)dut->rdata_o / 4096.0;
    worst = std::max(worst, std::fabs(got - (double)out[i]));
    refmax = std::max(refmax, std::fabs((double)out[i]));
  }
  std::printf("worst |diff| %.3e  (max |out| %.3e, rel %.2e)\n", worst, refmax, worst / refmax);
  bool ok = (worst / refmax < 3e-3);
  std::printf("%s\n", ok ? "PASS  rope == golden apply_rope (fixed point)" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
