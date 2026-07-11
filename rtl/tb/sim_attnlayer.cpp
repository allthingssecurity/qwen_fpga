// Full attention decoder layer, end to end. Attention mixer (real RTL: Vmv q/k/v
// proj, Vsm softmax, Vmo o_proj) -> residual -> post-norm -> MLP (Vmv gate/up, Vsw
// swiglu, Vmd down) -> residual. Proven glue: QK-norm, RoPE, scores/context dots,
// output gate, post-norm, residual. Checked against golden layer output.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vmv.h"
#include "Vmo.h"
#include "Vmd.h"
#include "Vsm.h"
#include "Vsw.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }
static Vmv* mv; static Vmo* mo; static Vmd* md; static Vsm* sm; static Vsw* sw;
static void T(Vmv*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmo*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmd*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vsm*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vsw*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static double sig(double x){return 1.0/(1.0+std::exp(-x));}

template<class D> static void proj(D* d, const int8_t* w, const float* sw_, double acts,
                                   int rows, int IN, double outpow, std::vector<double>& y){
  const int LANES=8;
  for(int o=0;o<rows;o++){ d->mult=(int32_t)llround(acts*sw_[o]*std::pow(2.0,outpow));
    for(int wd=0;wd<IN/LANES;wd++){uint64_t word=0;for(int j=0;j<LANES;j++)word|=(uint64_t)(uint8_t)w[(size_t)o*IN+wd*LANES+j]<<(j*8);
      d->w_en=1;d->w_word=word;T(d); if(d->y_vld)y.push_back((double)(int16_t)d->y_q);} } d->w_en=0;
}

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  mv=new Vmv;mo=new Vmo;md=new Vmd;sm=new Vsm;sw=new Vsw;
  std::ifstream f("artifacts/tv_attnlayer.bin",std::ios::binary);
  if(!f){std::fprintf(stderr,"run scripts/export_attnlayer_tv.py first\n");return 2;}
  int32_t hd[5]; f.read((char*)hd,20);
  const int TW=hd[0],HD=hd[1],NH=hd[2],NKV=hd[3],LUTN=hd[4],HID=1024,IN2=2048,INTER=3584,RD=64,LANES=8;
  auto rv=[&](int n){std::vector<float>x(n);f.read((char*)x.data(),4*n);return x;};
  auto rw=[&](int rows,int in){std::vector<int8_t>w((size_t)rows*in);f.read((char*)w.data(),(size_t)rows*in);return w;};
  auto lex=rv(LUTN),lsg=rv(LUTN),cos=rv(RD),sin=rv(RD),h_in=rv(HID);
  std::vector<int8_t> xi8(HID);f.read((char*)xi8.data(),HID);float sx;f.read((char*)&sx,4);
  auto wq=rw(4096,HID); auto swq=rv(4096);   // export interleaves weight,scale per projection
  auto wk_=rw(512,HID); auto swk=rv(512);
  auto wv_=rw(512,HID); auto swv=rv(512);
  auto wo=rw(HID,IN2); auto swo=rv(HID);
  auto qn=rv(HD),kn=rv(HD); auto kc=rv(TW*NKV*HD),vc=rv(TW*NKV*HD); auto pln=rv(HID);
  auto wg=rw(INTER,HID); auto swg=rv(INTER); auto wu=rw(INTER,HID); auto swu=rv(INTER); auto wd=rw(HID,INTER); auto swd=rv(HID);
  auto mixg=rv(HID); auto hn2g=rv(HID); auto hout=rv(HID);

  // ===== attention mixer (verified attnfull datapath) =====
  mv->rst=1;mv->load_en=0;mv->w_en=0;T(mv);T(mv);mv->rst=0;
  for(int i=0;i<HID;i++){mv->load_en=1;mv->x_byte=xi8[i];T(mv);}mv->load_en=0;
  std::vector<double> qg,kf,vf;
  proj(mv,wq.data(),swq.data(),(double)sx,4096,HID,24,qg);  // Q8
  proj(mv,wk_.data(),swk.data(),(double)sx,512,HID,24,kf);
  proj(mv,wv_.data(),swv.data(),(double)sx,512,HID,24,vf);
  for(double&x:qg)x/=256.0; for(double&x:kf)x/=256.0; for(double&x:vf)x/=256.0;
  auto rms=[&](double*x,const float*w,int n){double v=0;for(int i=0;i<n;i++)v+=x[i]*x[i];v/=n;double inv=1.0/std::sqrt(v+1e-6);for(int i=0;i<n;i++)x[i]=x[i]*inv*(1.0+(double)w[i]);};
  auto rope=[&](double*x){double r[64];for(int i=0;i<RD;i++)r[i]=x[i];for(int i=0;i<RD;i++){double h=(i<RD/2)?-r[i+RD/2]:r[i-RD/2];x[i]=r[i]*cos[i]+h*sin[i];}};
  std::vector<double> q(NH*HD),gate(NH*HD);
  for(int h=0;h<NH;h++){for(int d=0;d<HD;d++){q[h*HD+d]=qg[h*HD*2+d];gate[h*HD+d]=qg[h*HD*2+HD+d];}rms(&q[h*HD],qn.data(),HD);rope(&q[h*HD]);}
  std::vector<double> kcur(NKV*HD),vcur(NKV*HD);
  for(int kv=0;kv<NKV;kv++){for(int d=0;d<HD;d++){kcur[kv*HD+d]=kf[kv*HD+d];vcur[kv*HD+d]=vf[kv*HD+d];}rms(&kcur[kv*HD],kn.data(),HD);rope(&kcur[kv*HD]);}
  int Tn=TW+1;
  auto Kc=[&](int t,int kv,int d){return t<TW?(double)kc[((size_t)t*NKV+kv)*HD+d]:kcur[kv*HD+d];};
  auto Vc=[&](int t,int kv,int d){return t<TW?(double)vc[((size_t)t*NKV+kv)*HD+d]:vcur[kv*HD+d];};
  double scale=1.0/std::sqrt((double)HD);
  std::vector<double> o(NH*HD,0.0);
  sm->rst=1;sm->start=0;sm->we_lut=0;sm->we_s=0;T(sm);T(sm);sm->rst=0;
  for(int i=0;i<LUTN;i++){sm->we_lut=1;sm->addr_lut=i;sm->din_lut=(uint32_t)llround((double)lex[i]*65536.0);T(sm);}sm->we_lut=0;
  for(int h=0;h<NH;h++){int kv=h/(NH/NKV);
    for(int t=0;t<Tn;t++){double s=0;for(int d=0;d<HD;d++)s+=q[h*HD+d]*Kc(t,kv,d);s*=scale;
      sm->we_s=1;sm->addr_s=t;sm->din_s=(int32_t)llround(s*1024.0);T(sm);}sm->we_s=0;
    sm->n_scores=Tn;sm->start=1;T(sm);sm->start=0;int g=0;while(!sm->done&&g++<5000)T(sm);
    std::vector<double> w(Tn);for(int t=0;t<Tn;t++){sm->raddr_w=t;sm->eval();w[t]=(double)(uint32_t)sm->rdata_w/65536.0;}
    for(int d=0;d<HD;d++){double c=0;for(int t=0;t<Tn;t++)c+=w[t]*Vc(t,kv,d);o[h*HD+d]=c*sig(gate[h*HD+d]);}}
  double am=0;for(double x:o)am=std::max(am,std::fabs(x));double so=(am>0?am/127.0:1.0);
  std::vector<int8_t> oi8(IN2);for(int i=0;i<IN2;i++)oi8[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(o[i]/so)));
  mo->rst=1;mo->load_en=0;mo->w_en=0;T(mo);T(mo);mo->rst=0;
  for(int i=0;i<IN2;i++){mo->load_en=1;mo->x_byte=oi8[i];T(mo);}mo->load_en=0;
  std::vector<double> mixraw; proj(mo,wo.data(),swo.data(),so,HID,IN2,30,mixraw); // Q14

  // ===== residual, post-norm, MLP, residual =====
  std::vector<double> h1(HID); for(int i=0;i<HID;i++)h1[i]=(double)h_in[i]+mixraw[i]/16384.0;
  double var=0;for(double x:h1)var+=x*x;var/=HID;double inv=1.0/std::sqrt(var+1e-6);
  std::vector<double> hn2(HID);for(int i=0;i<HID;i++)hn2[i]=h1[i]*inv*(1.0+(double)pln[i]);
  double hm=0;for(double x:hn2)hm=std::max(hm,std::fabs(x));double shn=(hm>0?hm/127.0:1.0);
  std::vector<int8_t> hn2i8(HID);for(int i=0;i<HID;i++)hn2i8[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(hn2[i]/shn)));
  mv->rst=1;mv->load_en=0;mv->w_en=0;T(mv);T(mv);mv->rst=0;
  for(int i=0;i<HID;i++){mv->load_en=1;mv->x_byte=hn2i8[i];T(mv);}mv->load_en=0;
  std::vector<double> gt,up; proj(mv,wg.data(),swg.data(),shn,INTER,HID,28,gt); proj(mv,wu.data(),swu.data(),shn,INTER,HID,28,up);
  sw->rst=1;sw->in_vld=0;sw->we_lut=0;T(sw);T(sw);sw->rst=0;
  for(int i=0;i<LUTN;i++){double xv=(i-512)/64.0,sgv=1.0/(1.0+std::exp(-xv));sw->we_lut=1;sw->addr_lut=i;sw->din_lut=(uint32_t)llround(sgv*65536.0);T(sw);}sw->we_lut=0;
  std::vector<double> gg;
  for(int i=0;i<INTER;i++){sw->in_vld=1;sw->g=(int32_t)gt[i];sw->u=(int32_t)up[i];T(sw);if(sw->out_vld)gg.push_back((double)(int32_t)sw->y/4096.0);}
  sw->in_vld=0;T(sw);if(sw->out_vld)gg.push_back((double)(int32_t)sw->y/4096.0);
  double gmx=0;for(double x:gg)gmx=std::max(gmx,std::fabs(x));double sg=(gmx>0?gmx/127.0:1.0);
  std::vector<int8_t> gi8(INTER);for(int i=0;i<INTER;i++)gi8[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(gg[i]/sg)));
  md->rst=1;md->load_en=0;md->w_en=0;T(md);T(md);md->rst=0;
  for(int i=0;i<INTER;i++){md->load_en=1;md->x_byte=gi8[i];T(md);}md->load_en=0;
  std::vector<double> dnraw; proj(md,wd.data(),swd.data(),sg,HID,INTER,30,dnraw); // Q14
  std::vector<double> out(HID);for(int i=0;i<HID;i++)out[i]=h1[i]+dnraw[i]/16384.0;

  double mixe=0,mrf=0,hn2e=0,hrf=0;   // stage checkpoints vs golden
  for(int i=0;i<HID;i++){mixe=std::max(mixe,std::fabs(mixraw[i]/16384.0-(double)mixg[i]));mrf=std::max(mrf,std::fabs((double)mixg[i]));
                         hn2e=std::max(hn2e,std::fabs(hn2[i]-(double)hn2g[i]));hrf=std::max(hrf,std::fabs((double)hn2g[i]));}
  std::printf("  stage: attention mixer rel %.2e,  post-norm rel %.2e\n",mixe/mrf,hn2e/hrf);
  double worst=0,refmax=0;
  for(int i=0;i<HID;i++){worst=std::max(worst,std::fabs(out[i]-(double)hout[i]));refmax=std::max(refmax,std::fabs((double)hout[i]));}
  std::printf("FULL attention layer (mixer -> residual -> post-norm -> MLP -> residual):\n");
  std::printf("  out worst |diff| %.3e  (max %.3e, rel %.2e)\n", worst, refmax, worst/refmax);
  bool ok=(worst/refmax<6e-2);
  std::printf("%s\n", ok?"PASS  full attention LAYER == golden, end to end":"FAIL");
  return ok?0:1;
}
