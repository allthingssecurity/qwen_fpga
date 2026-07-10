// csim testbench: run the HLS delta rule against numpy golden vectors.
//
//   python3 scripts/export_testvectors.py
//   make -C hls csim
//
// Checks both the output activations and the evolved recurrent state. The state
// check is the one that matters -- an error there compounds over every future
// token, and a cold-start test would never see it.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "../src/deltanet.hpp"

using qwen::H;
using qwen::K;
using qwen::V;

static std::vector<float> slurp(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "cannot open %s -- run scripts/export_testvectors.py first\n", path);
    std::exit(2);
  }
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<float> buf(n / sizeof(float));
  f.read(reinterpret_cast<char*>(buf.data()), n);
  return buf;
}

struct Stat {
  double max_abs = 0.0, max_rel = 0.0;
  int argmax = -1;
};

static Stat compare(const float* got, const float* exp, size_t n) {
  Stat s;
  for (size_t i = 0; i < n; ++i) {
    const double d = std::fabs(double(got[i]) - double(exp[i]));
    if (d > s.max_abs) {
      s.max_abs = d;
      s.argmax = int(i);
    }
    const double denom = std::fabs(double(exp[i]));
    if (denom > 1e-6) s.max_rel = std::max(s.max_rel, d / denom);
  }
  return s;
}

int main() {
  const std::vector<float> tv = slurp("artifacts/tv_deltanet.bin");

  // layout must match scripts/export_testvectors.py
  size_t o = 0;
  const float* q = &tv[o]; o += H * K;
  const float* k = &tv[o]; o += H * K;
  const float* v = &tv[o]; o += H * V;
  const float* g = &tv[o]; o += H;
  const float* beta = &tv[o]; o += H;
  const float* S_in = &tv[o]; o += H * K * V;
  const float* o_ref = &tv[o]; o += H * V;
  const float* S_ref = &tv[o]; o += H * K * V;

  if (o != tv.size()) {
    std::fprintf(stderr, "test vector size mismatch: consumed %zu of %zu floats\n", o, tv.size());
    return 2;
  }

  static float S[H][K][V];
  static float out[H][V];
  std::memcpy(S, S_in, sizeof(S));

  qwen::delta_rule_layer(reinterpret_cast<const float(*)[K]>(q),
                         reinterpret_cast<const float(*)[K]>(k),
                         reinterpret_cast<const float(*)[V]>(v),
                         g, beta, S, out);

  const Stat so = compare(&out[0][0], o_ref, H * V);
  const Stat ss = compare(&S[0][0][0], S_ref, size_t(H) * K * V);

  std::printf("delta_rule_layer  H=%d K=%d V=%d\n", H, K, V);
  std::printf("  out  max|d| %.3e   max rel %.3e   (%d elems)\n", so.max_abs, so.max_rel, H * V);
  std::printf("  S'   max|d| %.3e   max rel %.3e   (%d elems)\n", ss.max_abs, ss.max_rel, H * K * V);

  // fp32 reassociation only: numpy contracts with BLAS, we contract in k-order.
  const double TOL = 2e-5;
  const bool ok = so.max_abs < TOL && ss.max_abs < TOL;
  std::printf("%s (tol %.0e)\n", ok ? "PASS" : "FAIL", TOL);
  return ok ? 0 : 1;
}
