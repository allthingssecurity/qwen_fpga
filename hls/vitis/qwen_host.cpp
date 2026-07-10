// XRT host for qwen_decode_kernel. Loads the packed int8 weight blob into HBM,
// feeds prompt tokens, generates greedily, prints ids.
//
// Builds on an AWS FPGA Developer AMI (or any host with XRT + the F2 platform):
//     g++ -std=c++17 qwen_host.cpp -o qwen_host \
//         -I$XILINX_XRT/include -L$XILINX_XRT/lib -lxrt_coreutil -pthread
//
// NOT COMPILED HERE -- no XRT/Vitis in the dev environment. This is the turnkey
// source; see docs/build_afi.md. Same driving logic as hls/tb/run_hls.cpp (which
// IS verified in csim), so behaviour is pinned to the checked datapath.
//
//     hw_emu:   XCL_EMULATION_MODE=hw_emu ./qwen_host qwen.hw_emu.xclbin blob.bin ids.txt
//     hardware: ./qwen_host qwen.awsxclbin qwen35_int8.bin ids.txt

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

static constexpr int VOCAB = 248320;
static constexpr int N_FULL = 6;
static constexpr int KV_PROJ = 512;
static constexpr int MAX_CTX = 4096;
static constexpr int EOS = 248044;   // text_config eos_token_id

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <xclbin> <blob.bin> <ids.txt> [n_new]\n", argv[0]);
    return 2;
  }
  const std::string xclbin = argv[1], blobp = argv[2], idsp = argv[3];
  const int n_new = argc > 4 ? std::atoi(argv[4]) : 40;

  // load weight blob
  std::ifstream bf(blobp, std::ios::binary | std::ios::ate);
  const size_t blob_bytes = bf.tellg();
  bf.seekg(0);
  std::vector<uint8_t> blob(blob_bytes);
  bf.read(reinterpret_cast<char*>(blob.data()), blob_bytes);

  std::vector<int> prompt;
  { std::ifstream f(idsp); int t; while (f >> t) prompt.push_back(t); }

  // XRT setup
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(xclbin);
  auto kernel = xrt::kernel(device, uuid, "qwen_decode_kernel");

  // HBM buffers. The four weight views alias one physical blob; the .cfg sptags
  // place them on different pseudo-channels. group_id per arg picks the bank.
  auto bo_w = xrt::bo(device, blob_bytes, xrt::bo::flags::normal, kernel.group_id(0));
  bo_w.write(blob.data());
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto bo_kv = xrt::bo(device, (size_t)N_FULL * MAX_CTX * KV_PROJ * sizeof(float),
                       xrt::bo::flags::normal, kernel.group_id(4));
  auto bo_argmax = xrt::bo(device, sizeof(int), xrt::bo::flags::normal, kernel.group_id(6));
  auto bo_logits = xrt::bo(device, (size_t)VOCAB * sizeof(float),
                           xrt::bo::flags::normal, kernel.group_id(7));

  auto step = [&](int tok, int pos) -> int {
    auto run = kernel(bo_w, bo_w, bo_w, bo_w, bo_kv, pos, tok, bo_argmax, bo_logits);
    run.wait();
    bo_argmax.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int am = 0;
    bo_argmax.read(&am);
    return am;
  };

  int pos = 0, cur = 0;
  for (int t : prompt) { cur = step(t, pos); ++pos; }   // prefill
  std::vector<int> gen;
  for (int i = 0; i < n_new && cur != EOS; ++i) {
    gen.push_back(cur);
    cur = step(cur, pos);
    ++pos;
  }

  for (size_t i = 0; i < gen.size(); ++i)
    std::printf("%d%s", gen[i], i + 1 < gen.size() ? " " : "\n");
  return 0;
}
