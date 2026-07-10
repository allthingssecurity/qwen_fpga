// Run a prompt through the SYNTHESISABLE HLS decode (qwen_synth.hpp) -- the same
// C++ that goes to Vitis. Greedy autoregressive generation from int8 weights in
// the packed HBM blob. This is not the numpy reference; it is the hardware
// datapath, exercised on the CPU via csim.
//
//   host wrapper: run_hls.py tokenises the prompt and detokenises the output.
//   direct:  ./artifacts/run_hls weights.idx qwen35_int8.bin ids.txt n_new eos
//
// ids.txt: whitespace-separated prompt token ids. Prints generated ids to
// stdout; timing to stderr.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "../src/qwen_synth.hpp"

using namespace qwen;

static const uint8_t* map_blob(const char* path, size_t* len) {
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) { std::fprintf(stderr, "open %s failed\n", path); std::exit(2); }
  struct stat sb; ::fstat(fd, &sb); *len = sb.st_size;
  void* p = ::mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) { std::fprintf(stderr, "mmap failed\n"); std::exit(2); }
  return static_cast<const uint8_t*>(p);
}

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: run_hls <bin> <ids.txt> <n_new> [eos]\n"); return 2; }
  const char* binp = argv[1];
  const char* idsp = argv[2];
  const int n_new = std::atoi(argv[3]);
  const int eos = argc > 4 ? std::atoi(argv[4]) : -1;

  size_t blen = 0;
  const uint8_t* blob = map_blob(binp, &blen);

  std::vector<int> prompt;
  { std::ifstream f(idsp); int t; while (f >> t) prompt.push_back(t); }
  if (prompt.empty()) { std::fprintf(stderr, "no prompt ids\n"); return 2; }

  DecodeState* st = new DecodeState();
  st->reset();

  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  int cur = 0;
  for (size_t i = 0; i < prompt.size(); ++i) cur = qwen_top(blob, *st, prompt[i], nullptr);
  auto t1 = clk::now();

  std::vector<int> gen;
  for (int i = 0; i < n_new; ++i) {
    if (cur == eos) break;
    gen.push_back(cur);
    cur = qwen_top(blob, *st, cur, nullptr);
  }
  auto t2 = clk::now();

  for (size_t i = 0; i < gen.size(); ++i) std::printf("%d%s", gen[i], i + 1 < gen.size() ? " " : "\n");

  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const double pf = ms(t0, t1), dc = ms(t1, t2);
  std::fprintf(stderr,
               "[hls-csim] prefill %zu tok %.0f ms (%.1f tok/s) | decode %zu tok %.0f ms (%.1f tok/s)\n",
               prompt.size(), pf, prompt.size() / (pf / 1000.0),
               gen.size(), dc, gen.size() / (dc / 1000.0));
  delete st;
  return 0;
}
