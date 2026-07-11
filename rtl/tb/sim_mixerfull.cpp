// Full DeltaNet mixer, hidden in -> mixer out, end to end. Real RTL for the heavy
// compute: Vmv (in_proj int8 matvec), Vcv (conv), Vmix (gate+recurrence+gnorm, per
// head), Vmo (out_proj int8 matvec). Proven glue in the harness: silu, l2 norm,
// split. Checked against golden gated_deltanet output.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vmv.h"
#include "Vmo.h"
#include "Vcv.h"
#include "Vmix.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }
static Vmv* mv; static Vmo* mo; static Vcv* cv; static Vmix* mx;
static void tmv(){mv->clk=0;mv->eval();mv->clk=1;mv->eval();}
static void tmo(){mo->clk=0;mo->eval();mo->clk=1;mo->eval();}
static void tcv(){cv->clk=0;cv->eval();cv->clk=1;cv->eval();}
static void tmx(){mx->clk=0;mx->eval();mx->clk=1;mx->eval();}
static double silu(double x){ return x/(1.0+std::exp(-x)); }

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  mv=new Vmv; mo=new Vmo; cv=new Vcv; mx=new Vmix;
  const int H=16,K=128,CD=6144,HID=1024,LUTN=1024,IN2=2048,LANES=8;

  std::ifstream f("artifacts/tv_mixerfull.bin",std::ios::binary);
  if(!f){std::fprintf(stderr,"run scripts/export_mixerfull_tv.py first\n");return 2;}
  int32_t hd[5]; f.read((char*)hd,20);
  std::vector<float> lsp(LUTN),lex(LUTN),lsg(LUTN); f.read((char*)lsp.data(),4*LUTN);f.read((char*)lex.data(),4*LUTN);f.read((char*)lsg.data(),4*LUTN);
  std::vector<int8_t> xi8(HID); f.read((char*)xi8.data(),HID); float sx; f.read((char*)&sx,4);
  std::vector<int8_t> wqkv((size_t)CD*HID); f.read((char*)wqkv.data(),(size_t)CD*HID);
  std::vector<float> swq(CD); f.read((char*)swq.data(),4*CD);
  std::vector<float> cw(CD*4),cst(CD*4); f.read((char*)cw.data(),4*CD*4); f.read((char*)cst.data(),4*CD*4);
  std::vector<float> a(H),b(H),A(H),dt(H),z(H*K),nw(K); f.read((char*)a.data(),4*H);f.read((char*)b.data(),4*H);f.read((char*)A.data(),4*H);f.read((char*)dt.data(),4*H);f.read((char*)z.data(),4*H*K);f.read((char*)nw.data(),4*K);
  std::vector<float> rec((size_t)H*K*K); f.read((char*)rec.data(),4*(size_t)H*K*K);
  std::vector<int8_t> wout((size_t)HID*IN2); f.read((char*)wout.data(),(size_t)HID*IN2);
  std::vector<float> swo(HID); f.read((char*)swo.data(),4*HID);
  std::vector<float> og_ref(HID); f.read((char*)og_ref.data(),4*HID);

  auto QI=[](double x,double s){return (int32_t)llround(x*s);};

  // ---- stage 1: in_proj matvec -> qkv (Q10)
  mv->rst=1;mv->load_en=0;mv->w_en=0;tmv();tmv();mv->rst=0;
  for(int i=0;i<HID;i++){mv->load_en=1;mv->x_byte=xi8[i];tmv();} mv->load_en=0;
  std::vector<double> qkv; qkv.reserve(CD);
  for(int o=0;o<CD;o++){ mv->mult=(int32_t)llround((double)sx*swq[o]*std::pow(2.0,26));
    for(int w=0;w<HID/LANES;w++){uint64_t word=0;for(int j=0;j<LANES;j++)word|=(uint64_t)(uint8_t)wqkv[(size_t)o*HID+w*LANES+j]<<(j*8);
      mv->w_en=1;mv->w_word=word;tmv(); if(mv->y_vld)qkv.push_back((double)(int16_t)mv->y_q/1024.0);} }
  mv->w_en=0;

  // ---- stage 2: conv (Q10) -> stage 3: silu (glue)
  cv->rst=1;cv->in_vld=0;tcv();tcv();cv->rst=0;
  std::vector<double> qc(CD);
  int ci=0;
  for(int c=0;c<CD;c++){ cv->w0=QI(cst[c*4+1],1024);cv->w1=QI(cst[c*4+2],1024);cv->w2=QI(cst[c*4+3],1024);cv->w3=QI(qkv[c],1024);
    cv->c0=QI(cw[c*4+0],1024);cv->c1=QI(cw[c*4+1],1024);cv->c2=QI(cw[c*4+2],1024);cv->c3=QI(cw[c*4+3],1024);
    cv->in_vld=1;tcv(); if(cv->out_vld){ qc[ci]=silu((double)(int32_t)cv->y/1024.0); ci++; } }
  cv->in_vld=0;tcv(); if(cv->out_vld){ qc[ci]=silu((double)(int32_t)cv->y/1024.0); ci++; }

  // ---- stage 4: split + stage 5: l2norm (glue) -> q,k,v per head
  auto l2 = [&](double* x,int n,double scale){ double ss=0; for(int i=0;i<n;i++)ss+=x[i]*x[i];
    double inv=1.0/std::sqrt(ss+1e-6); for(int i=0;i<n;i++)x[i]=x[i]*inv*scale; };
  std::vector<double> q(H*K),k(H*K),v(H*K);
  for(int h=0;h<H;h++)for(int i=0;i<K;i++){ q[h*K+i]=qc[h*K+i]; k[h*K+i]=qc[2048+h*K+i]; v[h*K+i]=qc[4096+h*K+i]; }
  for(int h=0;h<H;h++){ l2(&q[h*K],K,1.0/std::sqrt((double)K)); l2(&k[h*K],K,1.0); }

  // ---- stage 6: mixer core (gate+recurrence+gnorm) per head -> og (Q16)
  mx->rst=1;mx->start=0;mx->we_gsp=mx->we_gex=mx->we_gsg=mx->we_s=mx->we_q=mx->we_k=mx->we_v=mx->we_nlut=mx->we_w=mx->we_z=0;
  tmx();tmx();mx->rst=0;
  for(int i=0;i<LUTN;i++){ mx->g_addr=i;
    mx->we_gsp=1;mx->g_din=(uint32_t)llround((double)lsp[i]*65536.0);tmx();mx->we_gsp=0;
    mx->we_gex=1;mx->g_din=(uint32_t)llround((double)lex[i]*1073741824.0);tmx();mx->we_gex=0;
    mx->we_gsg=1;mx->g_din=(uint32_t)llround((double)lsg[i]*16777216.0);tmx();mx->we_gsg=0;
    mx->we_nlut=1;mx->n_addr=i;mx->n_din=(uint32_t)llround((double)lsg[i]*65536.0);tmx();mx->we_nlut=0; }
  for(int i=0;i<K;i++){mx->we_w=1;mx->addr_wz=i;mx->din_w=QI(nw[i],65536);tmx();}mx->we_w=0;
  std::vector<double> og(H*K);
  for(int h=0;h<H;h++){
    mx->gm_a=QI(a[h],1024);mx->gm_dt=QI(dt[h],1024);mx->gm_A=QI(A[h],16384);mx->gm_b=QI(b[h],1024);
    for(int i=0;i<K;i++){mx->addr_qkv=i;mx->we_q=1;mx->din_q=QI(q[h*K+i],16777216);mx->we_k=1;mx->din_k=QI(k[h*K+i],16777216);mx->we_v=1;mx->din_v=QI(v[h*K+i],16777216);tmx();}
    mx->we_q=mx->we_k=mx->we_v=0;
    for(int i=0;i<K*K;i++){mx->we_s=1;mx->addr_s=i;mx->din_s=QI(rec[(size_t)h*K*K+i],16777216);tmx();}mx->we_s=0;
    for(int i=0;i<K;i++){mx->we_z=1;mx->addr_wz=i;mx->din_z=QI(z[h*K+i],4096);tmx();}mx->we_z=0;
    mx->start=1;tmx();mx->start=0; int g=0; while(!mx->done&&g++<200000)tmx();
    for(int i=0;i<K;i++){mx->raddr_y=i;mx->eval();og[h*K+i]=(double)(int32_t)mx->rdata_y/65536.0;}
  }

  // ---- stage 7: collect og[2048] -> int8 ; stage 8: out_proj matvec (Q14)
  double amax=0; for(double x:og)amax=std::max(amax,std::fabs(x)); double so=(amax>0?amax/127.0:1.0);
  std::vector<int8_t> ogi8(IN2); for(int i=0;i<IN2;i++)ogi8[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(og[i]/so)));
  mo->rst=1;mo->load_en=0;mo->w_en=0;tmo();tmo();mo->rst=0;
  for(int i=0;i<IN2;i++){mo->load_en=1;mo->x_byte=ogi8[i];tmo();}mo->load_en=0;
  std::vector<double> out; out.reserve(HID);
  for(int o=0;o<HID;o++){ mo->mult=(int32_t)llround(so*swo[o]*std::pow(2.0,30));   // Q14 out
    for(int w=0;w<IN2/LANES;w++){uint64_t word=0;for(int j=0;j<LANES;j++)word|=(uint64_t)(uint8_t)wout[(size_t)o*IN2+w*LANES+j]<<(j*8);
      mo->w_en=1;mo->w_word=word;tmo(); if(mo->y_vld)out.push_back((double)(int16_t)mo->y_q/16384.0);} }
  mo->w_en=0;

  // ---- verify vs golden mixer output
  int n=(int)out.size(); double worst=0,refmax=0;
  for(int i=0;i<HID&&i<n;i++){worst=std::max(worst,std::fabs(out[i]-(double)og_ref[i]));refmax=std::max(refmax,std::fabs((double)og_ref[i]));}
  std::printf("FULL DeltaNet mixer (hidden -> in_proj -> conv -> silu -> l2 -> recurrence -> out_proj):\n");
  std::printf("  produced %d of %d outputs\n", n, HID);
  std::printf("  out worst |diff| %.3e  (max %.3e, rel %.2e)\n", worst, refmax, worst/refmax);
  bool ok=(n==HID && worst/refmax<5e-2);
  std::printf("%s\n", ok?"PASS  full DeltaNet mixer == golden, end to end":"FAIL");
  return ok?0:1;
}
