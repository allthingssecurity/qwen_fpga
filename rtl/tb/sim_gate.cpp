// Verilator testbench for the DeltaNet gate math, vs golden gexp and beta.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vgate_math.h"
#include "verilated.h"

static Vgate_math* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vgate_math;

  std::ifstream f("artifacts/tv_gate.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_gate_tv.py first\n"); return 2; }
  int32_t hdr[2]; f.read((char*)hdr, 8);
  int NH = hdr[0], LUTN = hdr[1];
  std::vector<float> lsp(LUTN), lex(LUTN), lsg(LUTN),
                     a(NH), b(NH), A(NH), dt(NH), gexp(NH), beta(NH);
  f.read((char*)lsp.data(), 4*LUTN); f.read((char*)lex.data(), 4*LUTN); f.read((char*)lsg.data(), 4*LUTN);
  f.read((char*)a.data(), 4*NH); f.read((char*)b.data(), 4*NH);
  f.read((char*)A.data(), 4*NH); f.read((char*)dt.data(), 4*NH);
  f.read((char*)gexp.data(), 4*NH); f.read((char*)beta.data(), 4*NH);

  dut->rst = 1; dut->in_vld = 0; dut->we_sp = dut->we_ex = dut->we_sg = 0;
  tick(); tick(); dut->rst = 0;
  for (int i = 0; i < LUTN; i++) {
    dut->addr_lut = i;
    dut->we_sp = 1; dut->din_lut = (uint32_t)llround((double)lsp[i] * 65536.0);        tick();  // Q16
    dut->we_sp = 0; dut->we_ex = 1; dut->din_lut = (uint32_t)llround((double)lex[i] * 1073741824.0); tick();  // Q30
    dut->we_ex = 0; dut->we_sg = 1; dut->din_lut = (uint32_t)llround((double)lsg[i] * 16777216.0);   tick();  // Q24
    dut->we_sg = 0;
  }

  double gw = 0, bw = 0;
  std::vector<double> gg, bb;
  for (int h = 0; h < NH; h++) {
    dut->in_vld = 1;
    dut->a  = (int32_t)llround((double)a[h] * 1024.0);
    dut->dt = (int32_t)llround((double)dt[h] * 1024.0);
    dut->A  = (int32_t)llround((double)A[h] * 16384.0);
    dut->b  = (int32_t)llround((double)b[h] * 1024.0);
    tick();
    if (dut->out_vld) { gg.push_back((double)(uint32_t)dut->gexp / 1073741824.0);
                        bb.push_back((double)(uint32_t)dut->beta / 16777216.0); }
  }
  dut->in_vld = 0; tick();
  if (dut->out_vld) { gg.push_back((double)(uint32_t)dut->gexp / 1073741824.0);
                      bb.push_back((double)(uint32_t)dut->beta / 16777216.0); }

  for (int h = 0; h < NH && h < (int)gg.size(); h++) {
    gw = std::max(gw, std::fabs(gg[h] - (double)gexp[h]));
    bw = std::max(bw, std::fabs(bb[h] - (double)beta[h]));
  }
  std::printf("gexp worst |diff| %.3e   beta worst |diff| %.3e\n", gw, bw);
  bool ok = (gw < 3e-3) && (bw < 3e-3) && (int)gg.size() == NH;
  std::printf("%s\n", ok ? "PASS  gate_math == golden gexp,beta (fixed point)" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
