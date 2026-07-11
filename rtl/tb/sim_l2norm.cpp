#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vl2norm.h"
#include "verilated.h"

static Vl2norm* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }
static int32_t q16(double x) { return (int32_t)llround(x * 65536.0); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vl2norm;
  const int N = 128;
  std::ifstream f("artifacts/tv_l2norm.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_l2norm_tv.py first\n"); return 2; }
  std::vector<float> x(N), y(N);
  f.read((char*)x.data(), 4 * N); f.read((char*)y.data(), 4 * N);

  dut->rst = 1; dut->start = 0; dut->we_x = 0; tick(); tick(); dut->rst = 0;
  for (int i = 0; i < N; i++) { dut->we_x = 1; dut->addr_x = i; dut->din_x = q16(x[i]); tick(); } dut->we_x = 0;
  dut->scale = q16(1.0 / std::sqrt((double)N));
  dut->start = 1; tick(); dut->start = 0;
  int guard = 0; while (!dut->done && guard++ < 2000) tick();
  if (!dut->done) { std::printf("FAIL: never finished\n"); return 1; }

  double worst = 0, refmax = 0;
  for (int i = 0; i < N; i++) {
    dut->raddr_y = i; dut->eval();
    double got = (double)(int32_t)dut->rdata_y / 65536.0;
    worst = std::max(worst, std::fabs(got - (double)y[i]));
    refmax = std::max(refmax, std::fabs((double)y[i]));
  }
  std::printf("worst |diff| %.3e (max %.3e, rel %.2e)\n", worst, refmax, worst / refmax);
  bool ok = (worst / refmax < 5e-3);
  std::printf("%s\n", ok ? "PASS  l2norm == golden (fixed point)" : "FAIL");
  delete dut; return ok ? 0 : 1;
}
