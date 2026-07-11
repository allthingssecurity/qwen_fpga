// Full DeltaNet recurrence stage: run the proven single-head core over all 16
// heads, each with its own gate inputs, q,k,v, warm state, and z, and check every
// head's gated-norm output against golden. Same RTL top as sim_mixer, looped.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vdeltanet_mixer_core.h"
#include "verilated.h"

static Vdeltanet_mixer_core* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }
static int32_t Q(double x, double s) { return (int32_t)llround(x * s); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vdeltanet_mixer_core;

  std::ifstream f("artifacts/tv_mixer16.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_mixer16_tv.py first\n"); return 2; }
  int32_t hd[3]; f.read((char*)hd, 12);
  const int H = hd[0], K = hd[1], LUTN = hd[2], V = K;
  std::vector<float> lsp(LUTN), lex(LUTN), lsg(LUTN), a(H), b(H), A(H), dt(H),
      q(H*K), k(H*K), v(H*V), S((size_t)H*K*V), z(H*V), w(V), og(H*V);
  auto rd = [&](std::vector<float>& x){ f.read((char*)x.data(), 4*x.size()); };
  rd(lsp); rd(lex); rd(lsg); rd(a); rd(b); rd(A); rd(dt);
  rd(q); rd(k); rd(v); rd(S); rd(z); rd(w); rd(og);

  dut->rst = 1; dut->start = 0;
  dut->we_gsp=dut->we_gex=dut->we_gsg=dut->we_s=dut->we_q=dut->we_k=dut->we_v=0;
  dut->we_nlut=dut->we_w=dut->we_z=0;
  tick(); tick(); dut->rst = 0;

  // load LUTs and w once (they persist across runs)
  for (int i=0;i<LUTN;i++){ dut->g_addr=i;
    dut->we_gsp=1; dut->g_din=(uint32_t)llround((double)lsp[i]*65536.0);      tick(); dut->we_gsp=0;
    dut->we_gex=1; dut->g_din=(uint32_t)llround((double)lex[i]*1073741824.0); tick(); dut->we_gex=0;
    dut->we_gsg=1; dut->g_din=(uint32_t)llround((double)lsg[i]*16777216.0);   tick(); dut->we_gsg=0;
    dut->we_nlut=1; dut->n_addr=i; dut->n_din=(uint32_t)llround((double)lsg[i]*65536.0); tick(); dut->we_nlut=0; }
  for (int i=0;i<V;i++){ dut->we_w=1; dut->addr_wz=i; dut->din_w=Q(w[i],65536.0); tick(); } dut->we_w=0;

  double worst=0, refmax=0; int bad=0;
  for (int h=0; h<H; h++) {
    dut->gm_a=Q(a[h],1024); dut->gm_dt=Q(dt[h],1024); dut->gm_A=Q(A[h],16384); dut->gm_b=Q(b[h],1024);
    for (int i=0;i<K;i++){ dut->addr_qkv=i;
      dut->we_q=1; dut->din_q=Q(q[h*K+i],16777216.0);
      dut->we_k=1; dut->din_k=Q(k[h*K+i],16777216.0);
      dut->we_v=1; dut->din_v=Q(v[h*V+i],16777216.0); tick(); }
    dut->we_q=dut->we_k=dut->we_v=0;
    for (int i=0;i<K*V;i++){ dut->we_s=1; dut->addr_s=i; dut->din_s=Q(S[(size_t)h*K*V+i],16777216.0); tick(); } dut->we_s=0;
    for (int i=0;i<V;i++){ dut->we_z=1; dut->addr_wz=i; dut->din_z=Q(z[h*V+i],4096.0); tick(); } dut->we_z=0;

    dut->start=1; tick(); dut->start=0;
    int guard=0; while(!dut->done && guard++<200000) tick();
    if(!dut->done){ std::printf("FAIL: head %d never finished\n", h); return 1; }

    double hw=0;
    for (int i=0;i<V;i++){ dut->raddr_y=i; dut->eval();
      double got=(double)(int32_t)dut->rdata_y/65536.0;
      hw=std::max(hw,std::fabs(got-(double)og[h*V+i]));
      worst=std::max(worst,std::fabs(got-(double)og[h*V+i]));
      refmax=std::max(refmax,std::fabs((double)og[h*V+i])); }
    if (hw > 5e-2) bad++;
  }
  std::printf("all %d heads run.  worst |diff| %.3e (max |og| %.3e, rel %.2e)\n", H, worst, refmax, worst/refmax);
  bool ok = (bad==0 && worst/refmax < 2e-2);
  std::printf("%s\n", ok ? "PASS  full 16-head DeltaNet recurrence stage == golden" : "FAIL");
  delete dut; return ok?0:1;
}
