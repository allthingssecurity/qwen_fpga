#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vgated_norm.h"
#include "verilated.h"

static Vgated_norm* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vgated_norm;
  std::ifstream f("artifacts/tv_gnorm.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_gnorm_tv.py first\n"); return 2; }
  int32_t hdr[2]; f.read((char*)hdr, 8);
  int V = hdr[0], LUTN = hdr[1];
  std::vector<float> lut(LUTN), o(V), w(V), z(V), y(V);
  f.read((char*)lut.data(), 4*LUTN);
  f.read((char*)o.data(), 4*V); f.read((char*)w.data(), 4*V);
  f.read((char*)z.data(), 4*V); f.read((char*)y.data(), 4*V);

  dut->rst = 1; dut->start = 0; dut->we_lut = dut->we_o = dut->we_w = dut->we_z = 0;
  tick(); tick(); dut->rst = 0;
  for (int i = 0; i < LUTN; i++) { dut->we_lut=1; dut->addr_lut=i; dut->din_lut=(uint32_t)llround((double)lut[i]*65536.0); tick(); } dut->we_lut=0;
  for (int i = 0; i < V; i++) { dut->we_o=1; dut->addr_o=i; dut->din_o=(int32_t)llround((double)o[i]*16777216.0); tick(); } dut->we_o=0;  // Q24
  for (int i = 0; i < V; i++) { dut->we_w=1; dut->addr_w=i; dut->din_w=(int32_t)llround((double)w[i]*65536.0); tick(); } dut->we_w=0;    // Q16
  for (int i = 0; i < V; i++) { dut->we_z=1; dut->addr_z=i; dut->din_z=(int32_t)llround((double)z[i]*4096.0); tick(); } dut->we_z=0;     // Q12

  dut->start = 1; tick(); dut->start = 0;
  int guard = 0; while (!dut->done && guard++ < 3000) tick();
  if (!dut->done) { std::printf("FAIL: never finished\n"); return 1; }

  double worst = 0, refmax = 0;
  for (int i = 0; i < V; i++) {
    dut->raddr_y = i; dut->eval();
    double got = (double)(int32_t)dut->rdata_y / 65536.0;
    worst = std::max(worst, std::fabs(got - (double)y[i]));
    refmax = std::max(refmax, std::fabs((double)y[i]));
  }
  std::printf("worst |diff| %.3e (max %.3e, rel %.2e)\n", worst, refmax, worst / refmax);
  bool ok = (worst / refmax < 1e-2);
  std::printf("%s\n", ok ? "PASS  gated_norm == golden (fixed point)" : "FAIL");
  delete dut; return ok ? 0 : 1;
}
