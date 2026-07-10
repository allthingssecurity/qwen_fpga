// GENERATED. XRT host: fill 32 HBM banks, run, time, report GB/s.
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <vector>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
static constexpr int N = 32;
int main(int argc, char** argv){
  const char* xclbin = argv[1];
  unsigned words = argc>2 ? atoi(argv[2]) : (64u*1024*1024/64); // 64MB/bank
  unsigned iters = argc>3 ? atoi(argv[3]) : 16;
  auto dev = xrt::device(0); auto uuid = dev.load_xclbin(xclbin);
  auto k = xrt::kernel(dev, uuid, "hbm_bw");
  size_t bytes = (size_t)words*64;
  std::vector<xrt::bo> bo; 
  for(int i=0;i<N;++i){ auto b=xrt::bo(dev,bytes,xrt::bo::flags::normal,k.group_id(i));
    auto p=b.map<uint8_t*>(); for(size_t j=0;j<bytes;j+=4096) p[j]=j; b.sync(XCL_BO_SYNC_BO_TO_DEVICE); bo.push_back(b);} 
  auto cs=xrt::bo(dev,N*sizeof(uint64_t),xrt::bo::flags::normal,k.group_id(N+2));
  auto run=xrt::run(k);
  for(int i=0;i<N;++i) run.set_arg(i,bo[i]);
  run.set_arg(N,words); run.set_arg(N+1,iters); run.set_arg(N+2,cs);
  auto t0=std::chrono::steady_clock::now();
  run.start(); run.wait();
  auto t1=std::chrono::steady_clock::now();
  double s=std::chrono::duration<double>(t1-t0).count();
  double gb=(double)N*bytes*iters/1e9;
  printf("\n== REAL F2 HBM bandwidth ==\n");
  printf("ports %d  %.0f MB/bank  x%u iters  = %.1f GB moved\n", N, bytes/1e6, iters, gb);
  printf("time %.4f s\n", s);
  printf("ACHIEVED  %.1f GB/s\n", gb/s);
  printf("-> decode  %.0f tok/s  (758.6 MB/token)\n", (gb/s)*1e9/758.6e6);
  return 0;
}
