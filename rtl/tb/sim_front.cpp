// Front of the DeltaNet mixer, composed from real RTL: the int8 matvec (in_proj)
// feeds, after a requant, the causal conv. Drives Vmv (matvec_i8) and Vcv
// (conv1d_tap) in sequence over all 6144 channels and checks the conv output
// (pre-silu) against golden. Proves the int8-matmul path bridges into the conv.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vmv.h"
#include "Vcv.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }   // required for a manual (non --exe) build

static Vmv* mv; static Vcv* cv;
static void tmv() { mv->clk = 0; mv->eval(); mv->clk = 1; mv->eval(); }
static void tcv() { cv->clk = 0; cv->eval(); cv->clk = 1; cv->eval(); }
static int32_t q10(double x) { return (int32_t)llround(x * 1024.0); }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  mv = new Vmv; cv = new Vcv;
  const int N = 6144, IN = 1024, LANES = 8, WPR = IN / LANES;

  std::ifstream f("artifacts/tv_front.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "run scripts/export_front_tv.py first\n"); return 2; }
  int32_t hd[2]; f.read((char*)hd, 8);
  std::vector<int8_t> xi8(IN), wi8((size_t)N * IN);
  float sx; std::vector<float> sw(N), cw(N * 4), st(N * 4), pre(N);
  f.read((char*)xi8.data(), IN);
  f.read((char*)&sx, 4);
  f.read((char*)wi8.data(), (size_t)N * IN);
  f.read((char*)sw.data(), 4 * N);
  f.read((char*)cw.data(), 4 * N * 4);
  f.read((char*)st.data(), 4 * N * 4);
  f.read((char*)pre.data(), 4 * N);

  // ---- matvec: load activation, stream weights, collect qkv (Q14) per row
  mv->rst = 1; mv->load_en = 0; mv->w_en = 0; tmv(); tmv(); mv->rst = 0;
  for (int i = 0; i < IN; i++) { mv->load_en = 1; mv->x_byte = xi8[i]; tmv(); }
  mv->load_en = 0;

  std::vector<int32_t> qkv; qkv.reserve(N);
  // target Q10 output (range +/-32) since in_proj values reach ~14; mult encodes
  // OUTQ=10, SHIFT=16 -> 2^26. This also lands directly in the conv's Q10 format.
  for (int o = 0; o < N; o++) {
    double c = (double)sx * (double)sw[o];
    mv->mult = (int32_t)llround(c * std::pow(2.0, 26));
    for (int w = 0; w < WPR; w++) {
      uint64_t word = 0;
      for (int j = 0; j < LANES; j++)
        word |= (uint64_t)(uint8_t)wi8[(size_t)o * IN + w * LANES + j] << (j * 8);
      mv->w_en = 1; mv->w_word = word; tmv();
      if (mv->y_vld) qkv.push_back((int32_t)(int16_t)mv->y_q);   // Q10
    }
  }
  mv->w_en = 0;
  std::printf("matvec produced %zu of %d rows\n", qkv.size(), N);

  // ---- conv: window = [state1,state2,state3, requant(qkv)], per channel
  cv->rst = 1; cv->in_vld = 0; tcv(); tcv(); cv->rst = 0;
  std::vector<double> got; got.reserve(N);
  for (int c = 0; c < N; c++) {
    int32_t qk_q10 = qkv[c];                            // already Q10 from matvec
    cv->w0 = q10(st[c * 4 + 1]); cv->w1 = q10(st[c * 4 + 2]);
    cv->w2 = q10(st[c * 4 + 3]); cv->w3 = qk_q10;
    cv->c0 = q10(cw[c * 4 + 0]); cv->c1 = q10(cw[c * 4 + 1]);
    cv->c2 = q10(cw[c * 4 + 2]); cv->c3 = q10(cw[c * 4 + 3]);
    cv->in_vld = 1; tcv();
    if (cv->out_vld) got.push_back((double)(int32_t)cv->y / 1024.0);
  }
  cv->in_vld = 0; tcv();
  if (cv->out_vld) got.push_back((double)(int32_t)cv->y / 1024.0);

  int n = (int)got.size();
  double worst = 0, refmax = 0;
  for (int c = 0; c < N && c < n; c++) {
    worst = std::max(worst, std::fabs(got[c] - (double)pre[c]));
    refmax = std::max(refmax, std::fabs((double)pre[c]));
  }
  std::printf("front (matvec -> requant -> conv), %d channels:\n", n);
  std::printf("  conv_pre worst |diff| %.3e (max %.3e, rel %.2e)\n", worst, refmax, worst / refmax);
  bool ok = (n == N && worst / refmax < 1e-2);
  std::printf("%s\n", ok ? "PASS  DeltaNet front == golden (matvec+conv composed)" : "FAIL");
  delete mv; delete cv; return ok ? 0 : 1;
}
