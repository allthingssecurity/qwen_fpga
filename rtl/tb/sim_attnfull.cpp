// Full attention mixer, hidden in -> mixer out, end to end. Real RTL for the heavy
// stages: Vmv (q/k/v_proj int8 matvec), Vsm (softmax), Vmo (o_proj int8 matvec).
// Proven glue in the harness: QK-norm (rmsnorm), RoPE, the scores and context dot
// products, and the output gate. Checked against golden full_attention.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vmv.h"
#include "Vmo.h"
#include "Vsm.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }
static Vmv* mv; static Vmo* mo; static Vsm* sm;
static void tmv(){mv->clk=0;mv->eval();mv->clk=1;mv->eval();}
static void tmo(){mo->clk=0;mo->eval();mo->clk=1;mo->eval();}
static void tsm(){sm->clk=0;sm->eval();sm->clk=1;sm->eval();}
static double sig(double x){ return 1.0/(1.0+std::exp(-x)); }

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  mv=new Vmv; mo=new Vmo; sm=new Vsm;

  std::ifstream f("artifacts/tv_attnfull.bin",std::ios::binary);
  if(!f){std::fprintf(stderr,"run scripts/export_attnfull_tv.py first\n");return 2;}
  int32_t hd[5]; f.read((char*)hd,20);
  const int T=hd[0], HD=hd[1], NH=hd[2], NKV=hd[3], LUTN=hd[4], HID=1024, IN2=2048, LANES=8, RD=64;
  std::vector<float> lex(LUTN),lsg(LUTN),cos(RD),sin(RD); f.read((char*)lex.data(),4*LUTN);f.read((char*)lsg.data(),4*LUTN);f.read((char*)cos.data(),4*RD);f.read((char*)sin.data(),4*RD);
  std::vector<int8_t> xi8(HID); f.read((char*)xi8.data(),HID); float sx; f.read((char*)&sx,4);
  auto rdw=[&](int rows){ std::vector<int8_t> w((size_t)rows*HID); f.read((char*)w.data(),(size_t)rows*HID);
                          std::vector<float> s(rows); f.read((char*)s.data(),4*rows); return std::make_pair(w,s); };
  auto Wq=rdw(4096); auto Wk=rdw(512); auto Wv=rdw(512);
  std::vector<int8_t> wo((size_t)HID*IN2); f.read((char*)wo.data(),(size_t)HID*IN2);
  std::vector<float> swo(HID); f.read((char*)swo.data(),4*HID);
  std::vector<float> qn(HD),kn(HD); f.read((char*)qn.data(),4*HD);f.read((char*)kn.data(),4*HD);
  std::vector<float> kc(T*NKV*HD),vc(T*NKV*HD); f.read((char*)kc.data(),4*T*NKV*HD);f.read((char*)vc.data(),4*T*NKV*HD);
  std::vector<float> outg(HID); f.read((char*)outg.data(),4*HID);

  // ---- projections: q_proj(4096), k_proj(512), v_proj(512) with same loaded x ----
  auto runproj=[&](std::vector<int8_t>&w,std::vector<float>&sw,int rows,std::vector<double>&y){
    for(int o=0;o<rows;o++){ mv->mult=(int32_t)llround((double)sx*sw[o]*std::pow(2.0,24)); // Q8 (range +/-8)
      for(int wd=0;wd<HID/LANES;wd++){uint64_t word=0;for(int j=0;j<LANES;j++)word|=(uint64_t)(uint8_t)w[(size_t)o*HID+wd*LANES+j]<<(j*8);
        mv->w_en=1;mv->w_word=word;tmv(); if(mv->y_vld)y.push_back((double)(int16_t)mv->y_q/256.0);} } mv->w_en=0; };
  mv->rst=1;mv->load_en=0;mv->w_en=0;tmv();tmv();mv->rst=0;
  for(int i=0;i<HID;i++){mv->load_en=1;mv->x_byte=xi8[i];tmv();}mv->load_en=0;
  std::vector<double> qg,kf,vf; runproj(Wq.first,Wq.second,4096,qg); runproj(Wk.first,Wk.second,512,kf); runproj(Wv.first,Wv.second,512,vf);

  // ---- QK-norm + RoPE (glue) ----
  auto rms=[&](double*x,const float*w,int n){ double v=0;for(int i=0;i<n;i++)v+=x[i]*x[i]; v/=n; double inv=1.0/std::sqrt(v+1e-6);
    for(int i=0;i<n;i++)x[i]=x[i]*inv*(1.0+(double)w[i]); };
  auto rope=[&](double*x){ double r[64]; for(int i=0;i<RD;i++)r[i]=x[i];
    for(int i=0;i<RD;i++){ double h=(i<RD/2)?-r[i+RD/2]:r[i-RD/2]; x[i]=r[i]*cos[i]+h*sin[i]; } };
  std::vector<double> q(NH*HD),gate(NH*HD);
  for(int h=0;h<NH;h++){ for(int d=0;d<HD;d++){q[h*HD+d]=qg[h*HD*2+d]; gate[h*HD+d]=qg[h*HD*2+HD+d];} rms(&q[h*HD],qn.data(),HD); rope(&q[h*HD]); }
  std::vector<double> kcur(NKV*HD),vcur(NKV*HD);
  for(int kv=0;kv<NKV;kv++){ for(int d=0;d<HD;d++){kcur[kv*HD+d]=kf[kv*HD+d]; vcur[kv*HD+d]=vf[kv*HD+d];} rms(&kcur[kv*HD],kn.data(),HD); rope(&kcur[kv*HD]); }

  // ---- build cache (warm + current), scores, softmax (RTL), context, gate ----
  int Tn=T+1;                                  // positions to attend over
  auto Kc=[&](int t,int kv,int d){ return t<T? (double)kc[((size_t)t*NKV+kv)*HD+d] : kcur[kv*HD+d]; };
  auto Vc=[&](int t,int kv,int d){ return t<T? (double)vc[((size_t)t*NKV+kv)*HD+d] : vcur[kv*HD+d]; };
  double scale=1.0/std::sqrt((double)HD);
  std::vector<double> o(NH*HD,0.0);
  sm->rst=1;sm->start=0;sm->we_lut=0;sm->we_s=0;tsm();tsm();sm->rst=0;
  for(int i=0;i<LUTN;i++){sm->we_lut=1;sm->addr_lut=i;sm->din_lut=(uint32_t)llround((double)lex[i]*65536.0);tsm();}sm->we_lut=0;
  for(int h=0;h<NH;h++){ int kv=h/(NH/NKV);
    // scores -> load into softmax (Q10)
    for(int t=0;t<Tn;t++){ double s=0; for(int d=0;d<HD;d++)s+=q[h*HD+d]*Kc(t,kv,d); s*=scale;
      sm->we_s=1;sm->addr_s=t;sm->din_s=(int32_t)llround(s*1024.0);tsm(); } sm->we_s=0;
    sm->n_scores=Tn; sm->start=1;tsm();sm->start=0; int g=0; while(!sm->done&&g++<5000)tsm();
    std::vector<double> w(Tn); for(int t=0;t<Tn;t++){sm->raddr_w=t;sm->eval(); w[t]=(double)(uint32_t)sm->rdata_w/65536.0;}
    // context + output gate
    for(int d=0;d<HD;d++){ double c=0; for(int t=0;t<Tn;t++)c+=w[t]*Vc(t,kv,d); o[h*HD+d]=c*sig(gate[h*HD+d]); }
  }

  // ---- o_proj matvec (Q14) ----
  double amax=0;for(double x:o)amax=std::max(amax,std::fabs(x)); double so=(amax>0?amax/127.0:1.0);
  std::vector<int8_t> oi8(IN2); for(int i=0;i<IN2;i++)oi8[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(o[i]/so)));
  mo->rst=1;mo->load_en=0;mo->w_en=0;tmo();tmo();mo->rst=0;
  for(int i=0;i<IN2;i++){mo->load_en=1;mo->x_byte=oi8[i];tmo();}mo->load_en=0;
  std::vector<double> out;
  for(int oo=0;oo<HID;oo++){ mo->mult=(int32_t)llround(so*swo[oo]*std::pow(2.0,30));
    for(int wd=0;wd<IN2/LANES;wd++){uint64_t word=0;for(int j=0;j<LANES;j++)word|=(uint64_t)(uint8_t)wo[(size_t)oo*IN2+wd*LANES+j]<<(j*8);
      mo->w_en=1;mo->w_word=word;tmo(); if(mo->y_vld)out.push_back((double)(int16_t)mo->y_q/16384.0);} } mo->w_en=0;

  int n=(int)out.size(); double worst=0,refmax=0;
  for(int i=0;i<HID&&i<n;i++){worst=std::max(worst,std::fabs(out[i]-(double)outg[i]));refmax=std::max(refmax,std::fabs((double)outg[i]));}
  std::printf("FULL attention mixer (hidden -> q/k/v_proj -> QKnorm -> RoPE -> scores -> softmax -> ctx -> gate -> o_proj):\n");
  std::printf("  produced %d of %d outputs\n", n, HID);
  std::printf("  out worst |diff| %.3e  (max %.3e, rel %.2e)\n", worst, refmax, worst/refmax);
  bool ok=(n==HID && worst/refmax<5e-2);
  std::printf("%s\n", ok?"PASS  full attention mixer == golden, end to end":"FAIL");
  return ok?0:1;
}
