// End-to-end test of the DeltaNet core composition (gate -> recurrence -> norm)
// for head 0, driven with real inputs and checked against golden.
//   gate inputs + LUTs           from artifacts/tv_gate.bin   (token 11751)
//   q,k,v (post-l2norm), state   from artifacts/tv_deltanet.bin (same token)
//   w, z, sigmoid LUT, golden y  from artifacts/tv_gnorm.bin   (same token)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vdeltanet_mixer_core.h"
#include "verilated.h"

static Vdeltanet_mixer_core* dut;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vdeltanet_mixer_core;
  const int K = 128, V = 128, LUTN = 1024, HH = 16;

  // ---- tv_gate: softplus,exp,sigmoid LUTs (fp32) + a,b,A,dt (head 0)
  std::ifstream fg("artifacts/tv_gate.bin", std::ios::binary);
  int32_t gh[2]; fg.read((char*)gh, 8);
  std::vector<float> lsp(LUTN), lex(LUTN), lsg(LUTN), a(HH), b(HH), A(HH), dt(HH);
  fg.read((char*)lsp.data(),4*LUTN); fg.read((char*)lex.data(),4*LUTN); fg.read((char*)lsg.data(),4*LUTN);
  fg.read((char*)a.data(),4*HH); fg.read((char*)b.data(),4*HH); fg.read((char*)A.data(),4*HH); fg.read((char*)dt.data(),4*HH);

  // ---- tv_deltanet: q,k,v (head 0), S_in (head 0)
  std::ifstream fd("artifacts/tv_deltanet.bin", std::ios::binary);
  std::vector<float> q(HH*K), k(HH*K), v(HH*V);
  fd.read((char*)q.data(),4*HH*K); fd.read((char*)k.data(),4*HH*K); fd.read((char*)v.data(),4*HH*V);
  fd.seekg(24704);  // S_in offset (from tv_deltanet.json)
  std::vector<float> S(K*V); fd.read((char*)S.data(), 4*K*V);   // head 0

  // ---- tv_gnorm: sigmoid LUT, w, z, golden y (head 0)
  std::ifstream fn("artifacts/tv_gnorm.bin", std::ios::binary);
  int32_t nh[2]; fn.read((char*)nh, 8);
  std::vector<float> nlut(LUTN), o_ignore(V), w(V), z(V), y(V);
  fn.read((char*)nlut.data(),4*LUTN); fn.read((char*)o_ignore.data(),4*V);
  fn.read((char*)w.data(),4*V); fn.read((char*)z.data(),4*V); fn.read((char*)y.data(),4*V);

  auto Q = [](double x, double s){ return (int32_t)llround(x*s); };

  dut->rst = 1; dut->start = 0;
  dut->we_gsp=dut->we_gex=dut->we_gsg=dut->we_s=dut->we_q=dut->we_k=dut->we_v=0;
  dut->we_nlut=dut->we_w=dut->we_z=0;
  tick(); tick(); dut->rst = 0;

  // load gate LUTs
  for (int i=0;i<LUTN;i++){ dut->g_addr=i;
    dut->we_gsp=1; dut->g_din=(uint32_t)llround((double)lsp[i]*65536.0);      tick(); dut->we_gsp=0;
    dut->we_gex=1; dut->g_din=(uint32_t)llround((double)lex[i]*1073741824.0); tick(); dut->we_gex=0;
    dut->we_gsg=1; dut->g_din=(uint32_t)llround((double)lsg[i]*16777216.0);   tick(); dut->we_gsg=0; }
  // gate head-0 inputs (held)
  dut->gm_a=Q(a[0],1024); dut->gm_dt=Q(dt[0],1024); dut->gm_A=Q(A[0],16384); dut->gm_b=Q(b[0],1024);

  // load recurrence q,k,v (head 0, Q24), S_in (Q24)
  for (int i=0;i<K;i++){ dut->addr_qkv=i;
    dut->we_q=1; dut->din_q=Q(q[i],16777216.0);
    dut->we_k=1; dut->din_k=Q(k[i],16777216.0);
    dut->we_v=1; dut->din_v=Q(v[i],16777216.0); tick(); }
  dut->we_q=dut->we_k=dut->we_v=0;
  for (int i=0;i<K*V;i++){ dut->we_s=1; dut->addr_s=i; dut->din_s=Q(S[i],16777216.0); tick(); } dut->we_s=0;

  // load gnorm sigmoid LUT (Q16), w (Q16), z (Q12)
  for (int i=0;i<LUTN;i++){ dut->we_nlut=1; dut->n_addr=i; dut->n_din=(uint32_t)llround((double)nlut[i]*65536.0); tick(); } dut->we_nlut=0;
  for (int i=0;i<V;i++){ dut->addr_wz=i; dut->we_w=1; dut->din_w=Q(w[i],65536.0); dut->we_z=1; dut->din_z=Q(z[i],4096.0); tick(); }
  dut->we_w=dut->we_z=0;

  // run the whole core
  dut->start=1; tick(); dut->start=0;
  int guard=0; while(!dut->done && guard++<200000) tick();
  if(!dut->done){ std::printf("FAIL: never finished\n"); return 1; }

  double worst=0, refmax=0;
  for(int i=0;i<V;i++){ dut->raddr_y=i; dut->eval();
    double got=(double)(int32_t)dut->rdata_y/65536.0;
    worst=std::max(worst,std::fabs(got-(double)y[i])); refmax=std::max(refmax,std::fabs((double)y[i])); }
  std::printf("DeltaNet core (gate->recurrence->norm), head 0:\n");
  std::printf("  final y worst |diff| %.3e  (max %.3e, rel %.2e)\n", worst, refmax, worst/refmax);
  bool ok = (worst/refmax < 2e-2);
  std::printf("%s\n", ok ? "PASS  full DeltaNet core == golden, end to end" : "FAIL");
  delete dut; return ok?0:1;
}
