// Verilator testbench for SwiGLU (silu via sigmoid LUT), vs golden silu(g)*u.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vswiglu.h"
#include "verilated.h"

static Vswiglu* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }
static int32_t q12(double x) { return (int32_t)llround(x * 4096.0); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vswiglu;

  std::ifstream f("artifacts/tv_swiglu.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_swiglu_tv.py first\n"); return 2; }
  int32_t hdr[2]; f.read((char*)hdr, 8);
  int N = hdr[0], LUTN = hdr[1];
  std::vector<float> lut(LUTN), g(N), u(N), out(N);
  f.read((char*)lut.data(), 4 * LUTN);
  f.read((char*)g.data(), 4 * N);
  f.read((char*)u.data(), 4 * N);
  f.read((char*)out.data(), 4 * N);

  dut->rst = 1; dut->in_vld = 0; dut->we_lut = 0;
  tick(); tick(); dut->rst = 0;
  for (int i = 0; i < LUTN; i++) {
    dut->we_lut = 1; dut->addr_lut = i;
    dut->din_lut = (uint32_t)llround((double)lut[i] * 65536.0);  // Q16
    tick();
  }
  dut->we_lut = 0;

  std::vector<double> got;
  got.reserve(N);
  for (int i = 0; i < N; i++) {
    dut->in_vld = 1; dut->g = q12(g[i]); dut->u = q12(u[i]);
    tick();
    if (dut->out_vld) got.push_back((double)(int32_t)dut->y / 4096.0);
  }
  dut->in_vld = 0; tick();
  if (dut->out_vld) got.push_back((double)(int32_t)dut->y / 4096.0);

  int n = (int)got.size();
  double worst = 0, refmax = 0;
  for (int i = 0; i < N && i < n; i++) {
    worst = std::max(worst, std::fabs(got[i] - (double)out[i]));
    refmax = std::max(refmax, std::fabs((double)out[i]));
  }
  std::printf("collected %d of %d, y worst |diff| %.3e (max %.3e, rel %.2e)\n",
              n, N, worst, refmax, worst / refmax);
  bool ok = (n == N && worst / refmax < 5e-3);
  std::printf("%s\n", ok ? "PASS  swiglu == golden silu(g)*u (fixed point)" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
