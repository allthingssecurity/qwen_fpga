// Verilator testbench for the fixed-point Gated DeltaNet head. Loads one head of
// the warm-state golden vectors, runs the recurrence, and checks the output o and
// the evolved state S against the fp32 model, within fixed-point tolerance.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vdeltanet_head.h"
#include "verilated.h"

static Vdeltanet_head* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

static int32_t q20(double x) { return (int32_t)llround(x * (double)(1 << 24)); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vdeltanet_head;
  const int H = 16, K = 128, V = 128, HEAD = 0;

  std::ifstream f("artifacts/tv_deltanet.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_testvectors.py first\n"); return 2; }
  std::vector<float> q(H*K), k(H*K), v(H*V), g(H), beta(H),
                     S_in(H*K*V), o_core(H*V), S_out(H*K*V);
  f.read((char*)q.data(), 4*H*K);
  f.read((char*)k.data(), 4*H*K);
  f.read((char*)v.data(), 4*H*V);
  f.read((char*)g.data(), 4*H);
  f.read((char*)beta.data(), 4*H);
  f.read((char*)S_in.data(), 4*H*K*V);
  f.read((char*)o_core.data(), 4*H*V);
  f.read((char*)S_out.data(), 4*H*K*V);

  const float* qh = &q[HEAD*K];
  const float* kh = &k[HEAD*K];
  const float* vh = &v[HEAD*V];
  const float* Sh = &S_in[(size_t)HEAD*K*V];
  const float* oref = &o_core[HEAD*V];
  const float* Sref = &S_out[(size_t)HEAD*K*V];

  dut->rst = 1; dut->start = 0;
  dut->we_s = dut->we_q = dut->we_k = dut->we_v = 0;
  tick(); tick(); dut->rst = 0;

  dut->gexp = (int32_t)llround(std::exp((double)g[HEAD]) * (double)(1u << 30));
  dut->beta = q20((double)beta[HEAD]);

  for (int i = 0; i < K; i++) { dut->we_q=1; dut->addr_q=i; dut->din_q=q20(qh[i]); tick(); } dut->we_q=0;
  for (int i = 0; i < K; i++) { dut->we_k=1; dut->addr_k=i; dut->din_k=q20(kh[i]); tick(); } dut->we_k=0;
  for (int i = 0; i < V; i++) { dut->we_v=1; dut->addr_v=i; dut->din_v=q20(vh[i]); tick(); } dut->we_v=0;
  for (int i = 0; i < K*V; i++) { dut->we_s=1; dut->addr_s=i; dut->din_s=q20(Sh[i]); tick(); } dut->we_s=0;

  dut->start = 1; tick(); dut->start = 0;
  int guard = 0;
  while (!dut->done && guard++ < 80000) tick();
  if (!dut->done) { std::printf("FAIL: never finished\n"); return 1; }

  // check o
  double o_worst = 0, o_ref_max = 0;
  for (int vv = 0; vv < V; vv++) {
    dut->raddr_o = vv; dut->eval();
    double got = (double)(int32_t)dut->rdata_o / (double)(1 << 24);
    o_worst = std::max(o_worst, std::fabs(got - (double)oref[vv]));
    o_ref_max = std::max(o_ref_max, std::fabs((double)oref[vv]));
  }
  // check evolved S
  double s_worst = 0, s_ref_max = 0;
  for (int i = 0; i < K*V; i++) {
    dut->raddr_s = i; dut->eval();
    double got = (double)(int32_t)dut->rdata_s / (double)(1 << 24);
    s_worst = std::max(s_worst, std::fabs(got - (double)Sref[i]));
    s_ref_max = std::max(s_ref_max, std::fabs((double)Sref[i]));
  }

  std::printf("o: worst |diff| %.3e  (max |o| %.3e, rel %.2e)\n", o_worst, o_ref_max, o_worst/o_ref_max);
  std::printf("S: worst |diff| %.3e  (max |S| %.3e, rel %.2e)\n", s_worst, s_ref_max, s_worst/s_ref_max);
  bool ok = (o_worst/o_ref_max < 2e-3) && (s_worst/s_ref_max < 2e-3);
  std::printf("%s\n", ok ? "PASS  deltanet_head == model (fixed point)" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
