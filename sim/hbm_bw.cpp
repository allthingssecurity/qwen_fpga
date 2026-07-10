// HBM bandwidth harness: drive the REAL Qwen3.5-0.8B decode access pattern
// through a DRAMsim3 model of the F2 HBM2 device, measure sustained GB/s, and
// derive tokens/sec. This answers the only question csim left open -- whether
// the 758 MB/token weight stream actually sustains ~390-460 GB/s from HBM.
//
// Not synthetic: the access ranges come from artifacts/access.txt, which is the
// exact byte layout pack_weights.py wrote and emit_access.py walked in kernel
// read order. Every int8 payload, every scale, every fp32 tensor, in order.
//
// The swept parameter is outstanding-request depth. HBM's address map keeps a
// sequential stream in one open row for ~1 KiB before striping to the next
// channel, so channel-level parallelism -- and thus achieved bandwidth -- is
// gated by how many reads are in flight. That depth is a real RTL knob (AXI
// read IDs / prefetch FIFO depth), so the bandwidth-vs-depth knee is the
// actionable result for the streamer design.
//
//   build: make -C sim
//   run:   ./sim/build/hbm_bw sim/F2_HBM2_16ch.ini artifacts/access.txt <depth>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "memory_system.h"

namespace {

constexpr double WEIGHT_BYTES_PER_TOKEN = 758.6e6;  // from pack_weights.py ledger

struct Range {
  uint64_t off, nbytes;
};

std::vector<Range> load_ranges(const char* path, uint64_t* total_bytes) {
  FILE* f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open %s -- run scripts/emit_access.py first\n", path);
    std::exit(2);
  }
  std::vector<Range> r;
  char line[256];
  uint64_t tot = 0;
  while (std::fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    unsigned long long o, n;
    if (std::sscanf(line, "%llu %llu", &o, &n) == 2) {
      r.push_back({o, n});
      tot += n;
    }
  }
  std::fclose(f);
  *total_bytes = tot;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cfg = argc > 1 ? argv[1] : "sim/F2_HBM2_16ch.ini";
  const char* access = argc > 2 ? argv[2] : "artifacts/access.txt";
  const int depth = argc > 3 ? std::atoi(argv[3]) : 64;

  uint64_t stream_bytes = 0;
  const std::vector<Range> ranges = load_ranges(access, &stream_bytes);

  // completion bookkeeping via callbacks
  uint64_t completed = 0;
  auto read_cb = [&completed](uint64_t) { ++completed; };
  auto write_cb = [](uint64_t) {};

  dramsim3::MemorySystem mem(cfg, "artifacts/dramsim3_out", read_cb, write_cb);
  const int txn_bytes = mem.GetBurstLength() * mem.GetBusBits() / 8;
  const double tCK_ns = mem.GetTCK();
  const double peak_gbps = double(txn_bytes) * 0.0;  // placeholder, computed below

  // count transactions: each range -> ceil over txn_bytes, 64 B-aligned
  auto txns_in = [&](const Range& r) {
    uint64_t start = r.off & ~uint64_t(txn_bytes - 1);
    uint64_t end = r.off + r.nbytes;
    return (end - start + txn_bytes - 1) / txn_bytes;
  };
  uint64_t total_txns = 0;
  for (const auto& r : ranges) total_txns += txns_in(r);

  // peak: channels * txn_bytes worth of bus, 2x for DDR, per tCK. Recover
  // channels from GetBusBits (per-channel bus) is not exposed, so read it from
  // the achieved-vs-cycles calibration instead. We print achieved; for peak we
  // use the documented config (16 ch x 128 bit x 2 / tCK).
  (void)peak_gbps;
  const double cfg_peak_gbps = 16.0 * (128.0 / 8.0) * 2.0 / (tCK_ns);  // bytes/ns = GB/s

  // incremental address generator over ranges
  size_t ri = 0;
  uint64_t cur = ranges.empty() ? 0 : (ranges[0].off & ~uint64_t(txn_bytes - 1));
  auto next_addr = [&](uint64_t* out) -> bool {
    while (ri < ranges.size()) {
      uint64_t end = ranges[ri].off + ranges[ri].nbytes;
      if (cur < end) {
        *out = cur;
        cur += txn_bytes;
        return true;
      }
      if (++ri < ranges.size()) cur = ranges[ri].off & ~uint64_t(txn_bytes - 1);
    }
    return false;
  };

  uint64_t issued = 0, cycles = 0;
  int outstanding = 0;
  uint64_t pending_addr = 0;
  bool have_pending = false;

  while (completed < total_txns) {
    // issue as many as the model and our depth budget allow this cycle
    while (outstanding < depth) {
      if (!have_pending) {
        if (!next_addr(&pending_addr)) break;
        have_pending = true;
      }
      if (!mem.WillAcceptTransaction(pending_addr, false)) break;
      mem.AddTransaction(pending_addr, false);
      have_pending = false;
      ++issued;
      ++outstanding;
    }
    mem.ClockTick();
    ++cycles;
    // drain completions credited by the callback since last tick
    static uint64_t seen = 0;
    outstanding -= int(completed - seen);
    seen = completed;

    if ((cycles & ((1u << 22) - 1)) == 0 && issued == total_txns && completed < total_txns) {
      // safety: all issued, just draining
    }
  }

  const double time_ns = double(cycles) * tCK_ns;
  const double bytes = double(completed) * txn_bytes;
  const double achieved_gbps = bytes / time_ns;              // bytes/ns == GB/s
  const double util = achieved_gbps / cfg_peak_gbps;
  const double tok_per_s = achieved_gbps * 1e9 / WEIGHT_BYTES_PER_TOKEN;

  std::printf("\n==== F2 HBM2 bandwidth, real Qwen3.5-0.8B decode stream ====\n");
  std::printf("config            %s\n", cfg.c_str());
  std::printf("outstanding depth %d\n", depth);
  std::printf("txn size          %d B   tCK %.3f ns   cfg peak %.1f GB/s\n",
              txn_bytes, tCK_ns, cfg_peak_gbps);
  std::printf("stream            %.1f MB over %llu txns (%.1f MB moved)\n",
              stream_bytes / 1e6, (unsigned long long)total_txns, bytes / 1e6);
  std::printf("sim cycles        %llu  ->  %.3f ms wall\n",
              (unsigned long long)cycles, time_ns / 1e6);
  std::printf("-----------------------------------------------------------\n");
  std::printf("ACHIEVED          %.1f GB/s   (%.1f%% of %.0f GB/s peak)\n",
              achieved_gbps, util * 100.0, cfg_peak_gbps);
  std::printf("DECODE RATE       %.0f tok/s   (%.1f MB/token)\n",
              tok_per_s, WEIGHT_BYTES_PER_TOKEN / 1e6);
  std::printf("===========================================================\n");

  mem.PrintStats();
  return 0;
}
