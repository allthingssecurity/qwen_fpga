// Verilator testbench for the depthwise causal conv tap MAC, vs golden conv_pre.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vconv1d_tap.h"
#include "verilated.h"

static Vconv1d_tap* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }
static int32_t q10(double x) { return (int32_t)llround(x * 1024.0); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vconv1d_tap;
  const int N = 6144;

  std::ifstream f("artifacts/tv_conv.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_conv_tv.py first\n"); return 2; }
  std::vector<float> nq(N), st(N * 4), cw(N * 4), pre(N);
  f.read((char*)nq.data(), 4 * N);
  f.read((char*)st.data(), 4 * N * 4);
  f.read((char*)cw.data(), 4 * N * 4);
  f.read((char*)pre.data(), 4 * N);

  dut->rst = 1; dut->in_vld = 0; tick(); tick(); dut->rst = 0;

  std::vector<double> got; got.reserve(N);
  for (int c = 0; c < N; c++) {
    // window = state[1], state[2], state[3], new
    dut->w0 = q10(st[c * 4 + 1]); dut->w1 = q10(st[c * 4 + 2]);
    dut->w2 = q10(st[c * 4 + 3]); dut->w3 = q10(nq[c]);
    dut->c0 = q10(cw[c * 4 + 0]); dut->c1 = q10(cw[c * 4 + 1]);
    dut->c2 = q10(cw[c * 4 + 2]); dut->c3 = q10(cw[c * 4 + 3]);
    dut->in_vld = 1; tick();
    if (dut->out_vld) got.push_back((double)(int32_t)dut->y / 1024.0);
  }
  dut->in_vld = 0; tick();
  if (dut->out_vld) got.push_back((double)(int32_t)dut->y / 1024.0);

  int n = (int)got.size();
  double worst = 0, refmax = 0;
  for (int c = 0; c < N && c < n; c++) {
    worst = std::max(worst, std::fabs(got[c] - (double)pre[c]));
    refmax = std::max(refmax, std::fabs((double)pre[c]));
  }
  std::printf("collected %d of %d, worst |diff| %.3e (max %.3e, rel %.2e)\n", n, N, worst, refmax, worst / refmax);
  bool ok = (n == N && worst / refmax < 5e-3);
  std::printf("%s\n", ok ? "PASS  conv1d_tap == golden (fixed point)" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
