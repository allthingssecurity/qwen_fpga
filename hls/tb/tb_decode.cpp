// End-to-end csim: run the full HLS decode datapath over the same tokens as the
// golden int8 model and confirm it produces the same argmax at every step.
//
// This exercises EVERY kernel -- embed, rmsnorm(x2 conventions), int8 GEMV,
// conv1d, gated delta rule, partial RoPE, gated GQA attention, SwiGLU, tied
// head -- composed exactly as the hardware would, reading the identical packed
// int8 weights the FPGA streams from HBM.
//
//   python3 scripts/pack_weights.py            # -> artifacts/qwen35_int8.bin
//   python3 scripts/emit_weight_index.py       # -> artifacts/weights.idx
//   python3 scripts/export_decode_ref.py       # -> artifacts/decode_ref.bin
//   make -C hls csim_decode
//
// PASS = argmax matches golden at all steps AND step-0 logits agree to fp
// reassociation tolerance (int8 quant is identical on both sides; only the
// summation order in GEMV differs).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "../src/decoder.hpp"
#include "../src/qwen_synth.hpp"

using namespace qwen;

static const uint8_t* map_blob(const char* path, size_t* len) {
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) { std::fprintf(stderr, "open %s failed\n", path); std::exit(2); }
  struct stat sb;
  ::fstat(fd, &sb);
  *len = sb.st_size;
  void* p = ::mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) { std::fprintf(stderr, "mmap %s failed\n", path); std::exit(2); }
  return static_cast<const uint8_t*>(p);
}

static WeightBlob load_index(const char* path, const uint8_t* base) {
  WeightBlob W;
  W.base = base;
  std::ifstream g(path);
  if (!g) { std::fprintf(stderr, "open %s failed -- run emit_weight_index.py\n", path); std::exit(2); }
  // line: name off nbytes dtype scale_off scale_nbytes out in
  std::string name;
  uint64_t off, nbytes, so, sn;
  int dtype, out, in;
  int count = 0;
  while (g >> name >> off >> nbytes >> dtype >> so >> sn >> out >> in) {
    Entry ee;
    ee.off = off; ee.scale_off = so; ee.dtype = dtype; ee.out = out; ee.in = in;
    W.idx[name] = ee;
    ++count;
  }
  std::printf("loaded %d weight entries\n", count);
  return W;
}

int main() {
  size_t blob_len = 0;
  const uint8_t* blob = map_blob("artifacts/qwen35_int8.bin", &blob_len);
  WeightBlob W = load_index("artifacts/weights.idx", blob);
  std::printf("blob %.1f MB mapped\n", blob_len / 1e6);

  // reference: n, fed[n], argmax[n], logits0[VOCAB]
  std::ifstream rf("artifacts/decode_ref.bin", std::ios::binary);
  if (!rf) { std::fprintf(stderr, "run scripts/export_decode_ref.py first\n"); return 2; }
  int32_t n = 0;
  rf.read(reinterpret_cast<char*>(&n), 4);
  std::vector<int32_t> fed(n), am_ref(n);
  rf.read(reinterpret_cast<char*>(fed.data()), 4L * n);
  rf.read(reinterpret_cast<char*>(am_ref.data()), 4L * n);
  std::vector<float> logits0_ref(VOCAB);
  rf.read(reinterpret_cast<char*>(logits0_ref.data()), 4L * VOCAB);

  DecodeState* st = new DecodeState();
  DecodeState* st2 = new DecodeState();
  st->reset();
  st2->reset();
  std::vector<float> logits0(VOCAB);

  std::printf("\n%4s %8s %9s %9s %9s %s\n", "step", "tok", "str", "synth", "golden", "");
  std::printf("------------------------------------------------------\n");
  int mism = 0;
  for (int i = 0; i < n; ++i) {
    // string-addressed reference and synthesisable-top must both match golden
    const int am = decode_token(W, *st, fed[i], i == 0 ? logits0.data() : nullptr);
    const int am2 = qwen_top(blob, *st2, fed[i], nullptr);
    const bool ok = (am == am_ref[i]) && (am2 == am_ref[i]);
    if (!ok) ++mism;
    std::printf("%4d %8d %9d %9d %9d %s\n", i, fed[i], am, am2, am_ref[i], ok ? "ok" : "MISMATCH");
  }
  std::printf("------------------------------------------------------\n");

  double maxabs = 0.0, sabs = 0.0;
  for (int v = 0; v < VOCAB; ++v) {
    const double d = std::fabs(double(logits0[v]) - double(logits0_ref[v]));
    maxabs = d > maxabs ? d : maxabs;
    sabs += d;
  }
  std::printf("step-0 logits: max|d| %.3e  mean|d| %.3e  (vs int8 golden)\n",
              maxabs, sabs / VOCAB);

  const bool pass = (mism == 0) && (maxabs < 5e-2);
  std::printf("\n%s  (%d/%d steps: string-top AND synth-top match golden)\n",
              pass ? "PASS - HLS decode == golden int8" : "FAIL", n - mism, n);
  delete st;
  delete st2;
  return pass ? 0 : 1;
}
