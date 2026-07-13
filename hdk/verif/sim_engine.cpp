// Drive qwen_matvec_engine over its AXI4-Lite bus exactly as the host will on
// silicon: write dims, activation, per-row multipliers, and a weight tile, pulse
// start, poll done, read the int16 results, and check them against the golden
// dequantised int8 matvec. Proves the AXI wrapper and sequencer before any AFI build.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "Vqwen_matvec_engine.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }
static Vqwen_matvec_engine* e;
static void tick(){ e->clk=0; e->eval(); e->clk=1; e->eval(); }

static void axi_w(uint32_t a, uint32_t d){
  // present AW and W on SEPARATE cycles (a real master may) to exercise the
  // independent-latch handshake; the same-cycle-only version never started on silicon
  e->s_bready=1;
  e->s_awaddr=a; e->s_awvalid=1;
  int g=0; while(!e->s_awready && g++<100) tick();
  tick(); e->s_awvalid=0;                    // AW accepted
  e->s_wdata=d; e->s_wstrb=0xF; e->s_wvalid=1;
  g=0; while(!e->s_wready && g++<100) tick();
  tick(); e->s_wvalid=0;                      // W accepted, a cycle later
  g=0; while(!e->s_bvalid && g++<100) tick();
  tick(); e->s_bready=0;                      // consume bvalid
}
static uint32_t axi_r(uint32_t a){
  e->s_araddr=a; e->s_arvalid=1; e->s_rready=1;
  int g=0; while(!e->s_arready && g++<100) tick();
  tick();                                   // read address accepted
  e->s_arvalid=0;
  g=0; while(!e->s_rvalid && g++<100) tick();
  uint32_t d=e->s_rdata; tick(); e->s_rready=0;
  return d;
}

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  e=new Vqwen_matvec_engine;
  e->rst_n=0; e->s_awvalid=e->s_wvalid=e->s_bready=e->s_arvalid=e->s_rready=0;
  tick(); tick(); e->rst_n=1; tick();

  const int IN=1024, OUT=64, LANES=1;
  std::vector<int8_t> x(IN), w((size_t)OUT*IN); std::vector<int32_t> mult(OUT);
  for(int i=0;i<IN;i++) x[i]=(int8_t)(((i*7+3)%127)-63);
  for(int o=0;o<OUT;o++){ for(int i=0;i<IN;i++) w[(size_t)o*IN+i]=(int8_t)(((o*13+i*5+1)%127)-63);
    mult[o]=200+o; }

  // golden: y[o] = ( (sum_i x*w)*mult + 2^15 ) >> 16, int16 (matches dequant SHIFT=16)
  std::vector<int16_t> gold(OUT);
  for(int o=0;o<OUT;o++){ int64_t acc=0; for(int i=0;i<IN;i++) acc+=(int64_t)x[i]*w[(size_t)o*IN+i];
    int64_t p=acc*(int64_t)mult[o] + (1LL<<15); gold[o]=(int16_t)(p>>16); }

  // load over AXI-Lite
  axi_w(0x00008, IN); axi_w(0x0000C, OUT);
  uint32_t rb=axi_r(0x00008);                 // readback confirms writes land
  if((int)rb!=IN){ std::printf("FAIL readback: r_in=%u expected %d\n", rb, IN); return 1; }
  for(int i=0;i<IN;i++)  axi_w(0x10000 + i*4, (uint32_t)(uint8_t)x[i]);
  for(int o=0;o<OUT;o++) axi_w(0x20000 + o*4, (uint32_t)mult[o]);
  for(int o=0;o<OUT;o++) for(int wd=0;wd<IN/LANES;wd++){
    uint32_t word=0; for(int j=0;j<LANES;j++) word|=(uint32_t)(uint8_t)w[(size_t)o*IN+wd*LANES+j]<<(j*8);
    axi_w(0x100000 + (o*(IN/LANES)+wd)*4, word);      // weights in the addr[20] region
  }
  // run
  axi_w(0x00000, 1);
  int g=0; while(!(axi_r(0x00004)&1) && g++<200000) {}
  // read + check
  int bad=0; int16_t sample=0;
  for(int o=0;o<OUT;o++){ int16_t y=(int16_t)(axi_r(0x80000 + o*4)&0xFFFF); if(o==0)sample=y; if(y!=gold[o])bad++; }
  std::printf("qwen_matvec_engine over AXI-Lite: %d/%d rows correct  (y[0]=%d golden=%d)\n",OUT-bad,OUT,sample,gold[0]);
  bool ok=(bad==0);
  std::printf("%s\n", ok?"PASS  engine matvec == golden, driven entirely over AXI-Lite":"FAIL");
  return ok?0:1;
}
