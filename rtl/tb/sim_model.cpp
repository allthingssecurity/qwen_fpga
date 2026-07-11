// Whole model, one decode step through all 24 layers on the real RTL blocks. The
// harness plays the part of the on-chip controller: embedding lookup, sequence the
// layers ([DeltaNet x3, attention] x6), thread the hidden state, final norm, tied
// output head. Every matvec, conv, recurrence, softmax, and swiglu is real RTL; the
// light elementwise steps (norms, silu, l2, residual, dot products) are proven glue.
// Streams weights layer by layer from tv_model.bin. Checks the hidden entering each
// layer against the golden trace, then the predicted token against golden.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "Vmv.h"
#include "Vmvw.h"
#include "Vmo.h"
#include "Vmd.h"
#include "Vcv.h"
#include "Vmix.h"
#include "Vsm.h"
#include "Vsw.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }
static Vmv* mv; static Vmvw* mvw; static Vmo* mo; static Vmd* md; static Vcv* cv; static Vmix* mx; static Vsm* sm; static Vsw* sw;
static void T(Vmv*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmvw*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmo*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmd*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vcv*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmix*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vsm*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vsw*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static double siluf(double x){return x/(1.0+std::exp(-x));}
static double sigf(double x){return 1.0/(1.0+std::exp(-x));}
static int32_t QI(double x,double s){return (int32_t)llround(x*s);}

static const int HID=1024,K=128,CD=6144,INTER=3584,IN2=2048,HD=256,NH=8,NKV=2,RD=64,LUTN=1024,LANES=8;

// -------- file reader --------
static std::ifstream F;
static std::vector<float> rf(int n){std::vector<float>x(n);F.read((char*)x.data(),4*n);return x;}
static std::vector<int8_t> rb(size_t n){std::vector<int8_t>x(n);F.read((char*)x.data(),n);return x;}
static int32_t ri(){int32_t v;F.read((char*)&v,4);return v;}
struct W8{std::vector<int8_t> w;std::vector<float> s;};
static W8 rw8(int rows,int in){W8 r;r.w=rb((size_t)rows*in);r.s=rf(rows);return r;}

// -------- generic int8 matvec on a loaded activation --------
static long g_sat=0;
template<class D> static void proj(D* d,const int8_t* w,const float* s,double acts,int rows,int IN,double pw,std::vector<double>& y){
  for(int o=0;o<rows;o++){ d->mult=(int32_t)llround(acts*s[o]*std::pow(2.0,pw));
    for(int wd=0;wd<IN/LANES;wd++){uint64_t word=0;for(int j=0;j<LANES;j++)word|=(uint64_t)(uint8_t)w[(size_t)o*IN+wd*LANES+j]<<(j*8);
      d->w_en=1;d->w_word=word;T(d); if(d->y_vld){int16_t yq=(int16_t)d->y_q; if(yq==32767||yq==-32768)g_sat++; y.push_back((double)yq);}} } d->w_en=0;
}
// wide matvec collector: reads the full int32 (Q20) output, for the in_proj path
template<class D> static void projw(D* d,const int8_t* w,const float* s,double acts,int rows,int IN,double pw,std::vector<double>& y){
  for(int o=0;o<rows;o++){ d->mult=(int32_t)llround(acts*s[o]*std::pow(2.0,pw));
    for(int wd=0;wd<IN/LANES;wd++){uint64_t word=0;for(int j=0;j<LANES;j++)word|=(uint64_t)(uint8_t)w[(size_t)o*IN+wd*LANES+j]<<(j*8);
      d->w_en=1;d->w_word=word;T(d); if(d->y_vld)y.push_back((double)(int32_t)d->y_q);} } d->w_en=0;
}
template<class D> static void loadx(D* d,const int8_t* xi8,int n){
  d->rst=1;d->load_en=0;d->w_en=0;T(d);T(d);d->rst=0;
  for(int i=0;i<n;i++){d->load_en=1;d->x_byte=xi8[i];T(d);}d->load_en=0;
}
static void quant(const std::vector<double>& x,std::vector<int8_t>& q,double& s){
  double m=0;for(double v:x)m=std::max(m,std::fabs(v));s=(m>0?m/127.0:1.0);
  q.resize(x.size());for(size_t i=0;i<x.size();i++)q[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(x[i]/s)));
}
static std::vector<double> rmsn(const std::vector<double>& x,const std::vector<float>& w,bool gain1){
  double v=0;for(double e:x)v+=e*e;v/=x.size();double inv=1.0/std::sqrt(v+1e-6);
  std::vector<double> y(x.size());for(size_t i=0;i<x.size();i++)y[i]=x[i]*inv*(gain1?(1.0+(double)w[i]):(double)w[i]);return y;
}

// LUTs, cos/sin, embed head (globals held for the whole run)
static std::vector<float> Lsp,Lex,Lsg,gcos,gsin;

static void load_mix_luts(const std::vector<float>& gnorm){
  for(int i=0;i<LUTN;i++){mx->g_addr=i;
    mx->we_gsp=1;mx->g_din=(uint32_t)llround((double)Lsp[i]*65536.0);T(mx);mx->we_gsp=0;
    mx->we_gex=1;mx->g_din=(uint32_t)llround((double)Lex[i]*1073741824.0);T(mx);mx->we_gex=0;
    mx->we_gsg=1;mx->g_din=(uint32_t)llround((double)Lsg[i]*16777216.0);T(mx);mx->we_gsg=0;
    mx->we_nlut=1;mx->n_addr=i;mx->n_din=(uint32_t)llround((double)Lsg[i]*65536.0);T(mx);mx->we_nlut=0;}
  for(int i=0;i<K;i++){mx->we_w=1;mx->addr_wz=i;mx->din_w=QI(gnorm[i],65536);T(mx);}mx->we_w=0;
}

// ================= DeltaNet layer =================
static std::vector<double> deltanet_layer(const std::vector<double>& h,const std::vector<float>& iln,const std::vector<float>& pln){
  W8 qkv_w=rw8(CD,HID), z_w=rw8(IN2,HID);
  auto a_w=rf(16*HID), b_w=rf(16*HID), conv_w=rf(CD*4), conv_st=rf(CD*4);
  auto A=rf(16), dt=rf(16), gnorm=rf(K); auto rec=rf((size_t)16*K*K);
  W8 out_w=rw8(HID,IN2), gate_w=rw8(INTER,HID), up_w=rw8(INTER,HID), down_w=rw8(HID,INTER);

  auto hn=rmsn(h,iln,true); std::vector<int8_t> xi8; double sx; quant(hn,xi8,sx);
  // in_proj on the wide matvec: Q20 int32 output. After l2norm the small heads
  // need the extra fractional bits; Q10 (int16) loses their direction. See docs.
  loadx(mvw,xi8.data(),HID);
  std::vector<double> qkvr,zr; projw(mvw,qkv_w.w.data(),qkv_w.s.data(),sx,CD,HID,26,qkvr); projw(mvw,z_w.w.data(),z_w.s.data(),sx,IN2,HID,26,zr);
  const double Q20=1048576.0;
  std::vector<double> qkv(CD),z(IN2); for(int i=0;i<CD;i++)qkv[i]=qkvr[i]/Q20; for(int i=0;i<IN2;i++)z[i]=zr[i]/Q20;
  std::vector<double> a(16),b(16);
  for(int o=0;o<16;o++){double sa=0,sb=0;for(int i=0;i<HID;i++){sa+=(double)a_w[o*HID+i]*hn[i];sb+=(double)b_w[o*HID+i]*hn[i];}a[o]=sa;b[o]=sb;}
  // conv + silu
  // conv window at Q20, per-channel weights at Q10 -> product Q30, shifted by Q=10 -> Q20 out
  cv->rst=1;cv->in_vld=0;T(cv);T(cv);cv->rst=0; std::vector<double> qc(CD);int ci=0;
  for(int c=0;c<CD;c++){cv->w0=QI(conv_st[c*4+1],Q20);cv->w1=QI(conv_st[c*4+2],Q20);cv->w2=QI(conv_st[c*4+3],Q20);cv->w3=QI(qkv[c],Q20);
    cv->c0=QI(conv_w[c*4+0],1024);cv->c1=QI(conv_w[c*4+1],1024);cv->c2=QI(conv_w[c*4+2],1024);cv->c3=QI(conv_w[c*4+3],1024);
    cv->in_vld=1;T(cv);if(cv->out_vld)qc[ci++]=siluf((double)(int32_t)cv->y/Q20);}
  cv->in_vld=0;T(cv);if(cv->out_vld&&ci<CD)qc[ci++]=siluf((double)(int32_t)cv->y/Q20);
  std::vector<double> q(16*K),k(16*K),v(16*K);
  for(int h2=0;h2<16;h2++)for(int i=0;i<K;i++){q[h2*K+i]=qc[h2*K+i];k[h2*K+i]=qc[2048+h2*K+i];v[h2*K+i]=qc[4096+h2*K+i];}
  auto l2=[&](double*x,double sc){double ss=0;for(int i=0;i<K;i++)ss+=x[i]*x[i];double inv=1.0/std::sqrt(ss+1e-6);for(int i=0;i<K;i++)x[i]*=inv*sc;};
  for(int h2=0;h2<16;h2++){l2(&q[h2*K],1.0/std::sqrt((double)K));l2(&k[h2*K],1.0);}
  load_mix_luts(gnorm);
  std::vector<double> og(16*K);
  for(int h2=0;h2<16;h2++){mx->gm_a=QI(a[h2],1024);mx->gm_dt=QI(dt[h2],1024);mx->gm_A=QI(A[h2],16384);mx->gm_b=QI(b[h2],1024);
    for(int i=0;i<K;i++){mx->addr_qkv=i;mx->we_q=1;mx->din_q=QI(q[h2*K+i],16777216);mx->we_k=1;mx->din_k=QI(k[h2*K+i],16777216);mx->we_v=1;mx->din_v=QI(v[h2*K+i],16777216);T(mx);}mx->we_q=mx->we_k=mx->we_v=0;
    for(int i=0;i<K*K;i++){mx->we_s=1;mx->addr_s=i;mx->din_s=QI(rec[(size_t)h2*K*K+i],16777216);T(mx);}mx->we_s=0;
    for(int i=0;i<K;i++){mx->we_z=1;mx->addr_wz=i;mx->din_z=QI(z[h2*K+i],4096);T(mx);}mx->we_z=0;
    mx->start=1;T(mx);mx->start=0;int g=0;while(!mx->done&&g++<200000)T(mx);
    for(int i=0;i<K;i++){mx->raddr_y=i;mx->eval();og[h2*K+i]=(double)(int32_t)mx->rdata_y/65536.0;}}
  std::vector<int8_t> ogi8; double so; quant(og,ogi8,so);
  loadx(mo,ogi8.data(),IN2); std::vector<double> mixr; proj(mo,out_w.w.data(),out_w.s.data(),so,HID,IN2,30,mixr);
  std::vector<double> h1(HID); for(int i=0;i<HID;i++)h1[i]=h[i]+mixr[i]/16384.0;
  // MLP
  auto hn2=rmsn(h1,pln,true); std::vector<int8_t> h2i8; double shn; quant(hn2,h2i8,shn);
  loadx(mv,h2i8.data(),HID); std::vector<double> gt,up; proj(mv,gate_w.w.data(),gate_w.s.data(),shn,INTER,HID,28,gt); proj(mv,up_w.w.data(),up_w.s.data(),shn,INTER,HID,28,up);
  std::vector<double> gg; for(int i=0;i<INTER;i++){sw->in_vld=1;sw->g=(int32_t)gt[i];sw->u=(int32_t)up[i];T(sw);if(sw->out_vld)gg.push_back((double)(int32_t)sw->y/4096.0);}
  sw->in_vld=0;T(sw);if(sw->out_vld)gg.push_back((double)(int32_t)sw->y/4096.0);
  std::vector<int8_t> gi8; double sg; quant(gg,gi8,sg);
  loadx(md,gi8.data(),INTER); std::vector<double> dn; proj(md,down_w.w.data(),down_w.s.data(),sg,HID,INTER,30,dn);
  std::vector<double> out(HID); for(int i=0;i<HID;i++)out[i]=h1[i]+dn[i]/16384.0; return out;
}

// ================= attention layer =================
static std::vector<double> attn_layer(const std::vector<double>& h,const std::vector<float>& iln,const std::vector<float>& pln,int Tw){
  W8 q_w=rw8(4096,HID), k_w=rw8(512,HID), v_w=rw8(512,HID), o_w=rw8(HID,IN2);
  auto qn=rf(HD), kn=rf(HD); auto kc=rf(Tw*NKV*HD), vc=rf(Tw*NKV*HD);
  W8 gate_w=rw8(INTER,HID), up_w=rw8(INTER,HID), down_w=rw8(HID,INTER);

  auto hn=rmsn(h,iln,true); std::vector<int8_t> xi8; double sx; quant(hn,xi8,sx);
  loadx(mv,xi8.data(),HID);
  std::vector<double> qg,kf,vf; proj(mv,q_w.w.data(),q_w.s.data(),sx,4096,HID,24,qg); proj(mv,k_w.w.data(),k_w.s.data(),sx,512,HID,24,kf); proj(mv,v_w.w.data(),v_w.s.data(),sx,512,HID,24,vf);
  for(double&x:qg)x/=256.0;for(double&x:kf)x/=256.0;for(double&x:vf)x/=256.0;
  auto rms=[&](double*x,const float*w){double vv=0;for(int i=0;i<HD;i++)vv+=x[i]*x[i];vv/=HD;double inv=1.0/std::sqrt(vv+1e-6);for(int i=0;i<HD;i++)x[i]=x[i]*inv*(1.0+(double)w[i]);};
  auto rope=[&](double*x){double r[64];for(int i=0;i<RD;i++)r[i]=x[i];for(int i=0;i<RD;i++){double hh=(i<RD/2)?-r[i+RD/2]:r[i-RD/2];x[i]=r[i]*gcos[i]+hh*gsin[i];}};
  std::vector<double> q(NH*HD),gate(NH*HD);
  for(int hh=0;hh<NH;hh++){for(int d=0;d<HD;d++){q[hh*HD+d]=qg[hh*HD*2+d];gate[hh*HD+d]=qg[hh*HD*2+HD+d];}rms(&q[hh*HD],qn.data());rope(&q[hh*HD]);}
  std::vector<double> kcur(NKV*HD),vcur(NKV*HD);
  for(int kv=0;kv<NKV;kv++){for(int d=0;d<HD;d++){kcur[kv*HD+d]=kf[kv*HD+d];vcur[kv*HD+d]=vf[kv*HD+d];}rms(&kcur[kv*HD],kn.data());rope(&kcur[kv*HD]);}
  int Tn=Tw+1;
  auto Kc=[&](int t,int kv,int d){return t<Tw?(double)kc[((size_t)t*NKV+kv)*HD+d]:kcur[kv*HD+d];};
  auto Vc=[&](int t,int kv,int d){return t<Tw?(double)vc[((size_t)t*NKV+kv)*HD+d]:vcur[kv*HD+d];};
  double scale=1.0/std::sqrt((double)HD); std::vector<double> o(NH*HD,0.0);
  for(int i=0;i<LUTN;i++){sm->we_lut=1;sm->addr_lut=i;sm->din_lut=(uint32_t)llround((double)Lex[i]*65536.0);T(sm);}sm->we_lut=0;
  for(int hh=0;hh<NH;hh++){int kv=hh/(NH/NKV);
    for(int t=0;t<Tn;t++){double s=0;for(int d=0;d<HD;d++)s+=q[hh*HD+d]*Kc(t,kv,d);s*=scale;sm->we_s=1;sm->addr_s=t;sm->din_s=(int32_t)llround(s*1024.0);T(sm);}sm->we_s=0;
    sm->n_scores=Tn;sm->start=1;T(sm);sm->start=0;int g=0;while(!sm->done&&g++<5000)T(sm);
    std::vector<double> w(Tn);for(int t=0;t<Tn;t++){sm->raddr_w=t;sm->eval();w[t]=(double)(uint32_t)sm->rdata_w/65536.0;}
    for(int d=0;d<HD;d++){double c=0;for(int t=0;t<Tn;t++)c+=w[t]*Vc(t,kv,d);o[hh*HD+d]=c*sigf(gate[hh*HD+d]);}}
  std::vector<int8_t> oi8; double so; quant(o,oi8,so);
  loadx(mo,oi8.data(),IN2); std::vector<double> mixr; proj(mo,o_w.w.data(),o_w.s.data(),so,HID,IN2,30,mixr);
  std::vector<double> h1(HID); for(int i=0;i<HID;i++)h1[i]=h[i]+mixr[i]/16384.0;
  auto hn2=rmsn(h1,pln,true); std::vector<int8_t> h2i8; double shn; quant(hn2,h2i8,shn);
  loadx(mv,h2i8.data(),HID); std::vector<double> gt,up; proj(mv,gate_w.w.data(),gate_w.s.data(),shn,INTER,HID,28,gt); proj(mv,up_w.w.data(),up_w.s.data(),shn,INTER,HID,28,up);
  std::vector<double> gg; for(int i=0;i<INTER;i++){sw->in_vld=1;sw->g=(int32_t)gt[i];sw->u=(int32_t)up[i];T(sw);if(sw->out_vld)gg.push_back((double)(int32_t)sw->y/4096.0);}
  sw->in_vld=0;T(sw);if(sw->out_vld)gg.push_back((double)(int32_t)sw->y/4096.0);
  std::vector<int8_t> gi8; double sg; quant(gg,gi8,sg);
  loadx(md,gi8.data(),INTER); std::vector<double> dn; proj(md,down_w.w.data(),down_w.s.data(),sg,HID,INTER,30,dn);
  std::vector<double> out(HID); for(int i=0;i<HID;i++)out[i]=h1[i]+dn[i]/16384.0; return out;
}

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  mv=new Vmv;mvw=new Vmvw;mo=new Vmo;md=new Vmd;cv=new Vcv;mx=new Vmix;sm=new Vsm;sw=new Vsw;
  F.open("artifacts/tv_model.bin",std::ios::binary);
  if(!F){std::fprintf(stderr,"run scripts/export_model_tv.py first\n");return 2;}
  int magic=ri(),N=ri(),hid=ri(),V=ri(),lutn=ri(),Tw=ri(),target=ri(),gargmax=ri();
  assert(magic==0x51574E00 && hid==HID && lutn==LUTN);
  Lsp=rf(LUTN);Lex=rf(LUTN);Lsg=rf(LUTN);
  auto final_norm=rf(HID); auto embrow=rf(HID); gcos=rf(RD);gsin=rf(RD);
  std::vector<int8_t> emb=rb((size_t)V*HID); auto es=rf(V);
  auto glogits=rf(V); std::vector<std::vector<float>> trace(N+1);
  for(int i=0;i<=N;i++)trace[i]=rf(HID);
  std::printf("model: %d layers, vocab %d, warm %d, target %d, golden argmax %d\n",N,V,Tw,target,gargmax);

  // one-time LUT loads that persist (softmax exp, swiglu sigmoid over [-8,8])
  sw->rst=1;sw->in_vld=0;T(sw);T(sw);sw->rst=0;
  for(int i=0;i<LUTN;i++){double xv=(i-512)/64.0;sw->we_lut=1;sw->addr_lut=i;sw->din_lut=(uint32_t)llround(sigf(xv)*65536.0);T(sw);}sw->we_lut=0;
  sm->rst=1;sm->start=0;sm->we_lut=0;sm->we_s=0;T(sm);T(sm);sm->rst=0;
  mx->rst=1;mx->start=0;mx->we_gsp=mx->we_gex=mx->we_gsg=mx->we_s=mx->we_q=mx->we_k=mx->we_v=mx->we_nlut=mx->we_w=mx->we_z=0;T(mx);T(mx);mx->rst=0;

  std::vector<double> h(HID); for(int i=0;i<HID;i++)h[i]=embrow[i];
  double worst_layer=0;
  for(int i=0;i<N;i++){
    int lm=ri(),kind=ri(); assert(lm==(0x4C000000|i));
    // check hidden entering this layer vs golden
    double e=0,r=0;for(int d=0;d<HID;d++){e=std::max(e,std::fabs(h[d]-(double)trace[i][d]));r=std::max(r,std::fabs((double)trace[i][d]));}
    worst_layer=std::max(worst_layer,e/r);
    auto iln=rf(HID),pln=rf(HID);
    long s0=g_sat;
    h = kind==0 ? deltanet_layer(h,iln,pln) : attn_layer(h,iln,pln,Tw);
    int em=ri(); assert(em==(0x4C000000|i));
    std::printf("  layer %2d %-9s h_in rel %.2e   matvec saturations %ld\n",i,kind==0?"deltanet":"attn",e/r,g_sat-s0);
  }
  // final norm + tied head
  auto hf=rmsn(h,final_norm,true);
  double fe=0,fr=0;for(int d=0;d<HID;d++){fe=std::max(fe,std::fabs(hf[d]-(double)trace[N][d]));fr=std::max(fr,std::fabs((double)trace[N][d]));}
  int argmax=0;double best=-1e30,bg=-1e30;
  for(int vv=0;vv<V;vv++){double s=0;const int8_t* row=&emb[(size_t)vv*HID];for(int d=0;d<HID;d++)s+=hf[d]*(double)row[d];s*=(double)es[vv];
    if(s>best){best=s;argmax=vv;} if((double)glogits[vv]>bg)bg=glogits[vv];}
  std::printf("final hidden rel %.2e (worst layer-input rel %.2e)\n",fe/fr,worst_layer);
  std::printf("RTL argmax token %d  (golden %d)   %s\n",argmax,gargmax,argmax==gargmax?"MATCH":"differ");
  bool ok=(argmax==gargmax);
  std::printf("%s\n", ok?"PASS  whole model runs end to end and predicts the golden token":"FAIL");
  return ok?0:1;
}
