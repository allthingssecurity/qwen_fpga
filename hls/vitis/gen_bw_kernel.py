"""Generate the F2 HBM bandwidth kernel: N m_axi ports, one per HBM pseudo-
channel, each streaming reads with deep bursts, run concurrently via DATAFLOW.
Measures real achieved HBM bandwidth on silicon -- the DRAMsim3 experiment, on
hardware. Emits kernel .cpp, XRT host .cpp, and the v++ connectivity .cfg.

  python3 gen_bw_kernel.py <N_PORTS> <CLK_MHZ>
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
CLK = int(sys.argv[2]) if len(sys.argv) > 2 else 300

# ---------------- kernel ----------------
k = []
k.append('// GENERATED. F2 HBM bandwidth kernel: %d ports, one per pseudo-channel.' % N)
k.append('#include <ap_int.h>')
k.append('#include <cstdint>')
k.append('typedef ap_uint<512> u512;   // 64 bytes / beat')
k.append('')
k.append('static void read_bank(const u512* in, unsigned words, unsigned iters, uint64_t* out) {')
k.append('  ap_uint<512> acc = 0;')
k.append('  for (unsigned r = 0; r < iters; ++r)')
k.append('    for (unsigned i = 0; i < words; ++i) {')
k.append('#pragma HLS PIPELINE II = 1')
k.append('      acc ^= in[i];   // xor keeps the reads from being optimised away')
k.append('    }')
k.append('  *out = (uint64_t)acc.range(63, 0);')
k.append('}')
k.append('')
k.append('extern "C" void hbm_bw(')
k.append(',\n'.join('    const u512* in%d' % i for i in range(N)) + ',')
k.append('    unsigned words, unsigned iters, uint64_t* csum) {')
for i in range(N):
    k.append('#pragma HLS INTERFACE m_axi port = in%d bundle = g%d offset = slave num_read_outstanding = 64 max_read_burst_length = 64' % (i, i))
k.append('#pragma HLS INTERFACE m_axi port = csum bundle = gout offset = slave')
k.append('#pragma HLS INTERFACE s_axilite port = words')
k.append('#pragma HLS INTERFACE s_axilite port = iters')
k.append('#pragma HLS INTERFACE s_axilite port = return')
k.append('#pragma HLS DATAFLOW')
for i in range(N):
    k.append('  read_bank(in%d, words, iters, &csum[%d]);' % (i, i))
k.append('}')
open('hbm_bw_kernel.cpp', 'w').write('\n'.join(k) + '\n')

# ---------------- connectivity cfg ----------------
c = ['# GENERATED. Bind each port to its own HBM pseudo-channel.',
     'platform=%s' % '$AWS_PLATFORM_202420_2',
     'kernel_frequency=%d' % CLK, '', '[connectivity]', 'nk=hbm_bw:1']
for i in range(N):
    c.append('sp=hbm_bw_1.in%d:HBM[%d]' % (i, i))
c.append('sp=hbm_bw_1.csum:DDR[0]')
open('hbm_bw.cfg', 'w').write('\n'.join(c) + '\n')

# ---------------- XRT host ----------------
h = []
h.append('// GENERATED. XRT host: fill %d HBM banks, run, time, report GB/s.' % N)
h.append('#include <cstdint>\n#include <cstdio>\n#include <chrono>\n#include <vector>')
h.append('#include "xrt/xrt_device.h"\n#include "xrt/xrt_kernel.h"\n#include "xrt/xrt_bo.h"')
h.append('static constexpr int N = %d;' % N)
h.append('int main(int argc, char** argv){')
h.append('  const char* xclbin = argv[1];')
h.append('  unsigned words = argc>2 ? atoi(argv[2]) : (64u*1024*1024/64); // 64MB/bank')
h.append('  unsigned iters = argc>3 ? atoi(argv[3]) : 16;')
h.append('  auto dev = xrt::device(0); auto uuid = dev.load_xclbin(xclbin);')
h.append('  auto k = xrt::kernel(dev, uuid, "hbm_bw");')
h.append('  size_t bytes = (size_t)words*64;')
h.append('  std::vector<xrt::bo> bo; ')
h.append('  for(int i=0;i<N;++i){ auto b=xrt::bo(dev,bytes,xrt::bo::flags::normal,k.group_id(i));')
h.append('    auto p=b.map<uint8_t*>(); for(size_t j=0;j<bytes;j+=4096) p[j]=j; b.sync(XCL_BO_SYNC_BO_TO_DEVICE); bo.push_back(b);} ')
h.append('  auto cs=xrt::bo(dev,N*sizeof(uint64_t),xrt::bo::flags::normal,k.group_id(N+2));')
h.append('  auto run=xrt::run(k);')
h.append('  for(int i=0;i<N;++i) run.set_arg(i,bo[i]);')
h.append('  run.set_arg(N,words); run.set_arg(N+1,iters); run.set_arg(N+2,cs);')
h.append('  auto t0=std::chrono::steady_clock::now();')
h.append('  run.start(); run.wait();')
h.append('  auto t1=std::chrono::steady_clock::now();')
h.append('  double s=std::chrono::duration<double>(t1-t0).count();')
h.append('  double gb=(double)N*bytes*iters/1e9;')
h.append('  printf("\\n== REAL F2 HBM bandwidth ==\\n");')
h.append('  printf("ports %d  %.0f MB/bank  x%u iters  = %.1f GB moved\\n", N, bytes/1e6, iters, gb);')
h.append('  printf("time %.4f s\\n", s);')
h.append('  printf("ACHIEVED  %.1f GB/s\\n", gb/s);')
h.append('  printf("-> decode  %.0f tok/s  (758.6 MB/token)\\n", (gb/s)*1e9/758.6e6);')
h.append('  return 0;')
h.append('}')
open('hbm_bw_host.cpp', 'w').write('\n'.join(h) + '\n')

print('generated hbm_bw_kernel.cpp, hbm_bw.cfg, hbm_bw_host.cpp for %d ports @ %d MHz' % (N, CLK))
