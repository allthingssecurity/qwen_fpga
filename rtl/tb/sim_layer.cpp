// Full DeltaNet decoder layer, end to end. Mixer (real RTL: Vmv in_proj, Vcv conv,
// Vmix recurrence core, Vmo out_proj) -> residual -> post-norm -> MLP (Vmv gate/up,
// Vsw swiglu, Vmd down) -> residual. Proven glue: input norm (folded into x_i8),
// post-norm, silu, l2norm, split, residual add. Checked against golden layer output.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Vmv.h"
#include "Vmo.h"
#include "Vmd.h"
#include "Vcv.h"
#include "Vmix.h"
#include "Vsw.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }
static Vmv* mv; static Vmo* mo; static Vmd* md; static Vcv* cv; static Vmix* mx; static Vsw* sw;
static void T(Vmv*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmo*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmd*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vcv*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vmix*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static void T(Vsw*d){d->clk=0;d->eval();d->clk=1;d->eval();}
static double siluf(double x){return x/(1.0+std::exp(-x));}
static int32_t QI(double x,double s){return (int32_t)llround(x*s);}

// stream an int8 matvec that already has its activation loaded; collect dequant rows
template<class D> static void proj(D* d, const int8_t* w, const float* sw_, double acts,
                                   int rows, int IN, double outpow, std::vector<double>& y){
  const int LANES=8;
  for(int o=0;o<rows;o++){ d->mult=(int32_t)llround(acts*sw_[o]*std::pow(2.0,outpow));
    for(int wd=0;wd<IN/LANES;wd++){uint64_t word=0;for(int j=0;j<LANES;j++)word|=(uint64_t)(uint8_t)w[(size_t)o*IN+wd*LANES+j]<<(j*8);
      d->w_en=1;d->w_word=word;T(d); if(d->y_vld)y.push_back((double)(int16_t)d->y_q);} } d->w_en=0;
}

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  mv=new Vmv;mo=new Vmo;md=new Vmd;cv=new Vcv;mx=new Vmix;sw=new Vsw;
  const int H=16,K=128,CD=6144,HID=1024,INTER=3584,LUTN=1024,IN2=2048,LANES=8;

  std::ifstream f("artifacts/tv_layer.bin",std::ios::binary);
  if(!f){std::fprintf(stderr,"run scripts/export_layer_tv.py first\n");return 2;}
  int32_t hd[6]; f.read((char*)hd,24);
  auto rv=[&](int n){std::vector<float>x(n);f.read((char*)x.data(),4*n);return x;};
  auto rw=[&](int rows,int in){std::vector<int8_t>w((size_t)rows*in);f.read((char*)w.data(),(size_t)rows*in);return w;};
  auto lsp=rv(LUTN),lex=rv(LUTN),lsg=rv(LUTN); auto h_in=rv(HID);
  std::vector<int8_t> xi8(HID);f.read((char*)xi8.data(),HID); float sx;f.read((char*)&sx,4);
  auto wqkv=rw(CD,HID); auto swq=rv(CD);
  auto cw=rv(CD*4); auto cst=rv(CD*4);
  auto a=rv(H),b=rv(H),A=rv(H),dt=rv(H),z=rv(H*K),nw=rv(K); auto rec=rv(H*K*K);
  auto wout=rw(HID,IN2); auto swo=rv(HID); auto pln=rv(HID);
  auto wg=rw(INTER,HID); auto swg=rv(INTER); auto wu=rw(INTER,HID); auto swu=rv(INTER);
  auto wd=rw(HID,INTER); auto swd=rv(HID); auto hout=rv(HID);   // down_proj [1024,3584]: scale len = HID

  // ===== MIXER (reuses the verified full-mixer datapath) =====
  mv->rst=1;mv->load_en=0;mv->w_en=0;T(mv);T(mv);mv->rst=0;
  for(int i=0;i<HID;i++){mv->load_en=1;mv->x_byte=xi8[i];T(mv);}mv->load_en=0;
  std::vector<double> qkvraw; proj(mv,wqkv.data(),swq.data(),(double)sx,CD,HID,26,qkvraw); // Q10
  std::vector<double> qkv(CD); for(int i=0;i<CD;i++)qkv[i]=qkvraw[i]/1024.0;
  cv->rst=1;cv->in_vld=0;T(cv);T(cv);cv->rst=0;
  std::vector<double> qc(CD); int ci=0;
  for(int c=0;c<CD;c++){cv->w0=QI(cst[c*4+1],1024);cv->w1=QI(cst[c*4+2],1024);cv->w2=QI(cst[c*4+3],1024);cv->w3=QI(qkv[c],1024);
    cv->c0=QI(cw[c*4+0],1024);cv->c1=QI(cw[c*4+1],1024);cv->c2=QI(cw[c*4+2],1024);cv->c3=QI(cw[c*4+3],1024);
    cv->in_vld=1;T(cv); if(cv->out_vld)qc[ci++]=siluf((double)(int32_t)cv->y/1024.0);}
  cv->in_vld=0;T(cv); if(cv->out_vld&&ci<CD)qc[ci++]=siluf((double)(int32_t)cv->y/1024.0);
  auto l2=[&](double*x,double sc){double ss=0;for(int i=0;i<K;i++)ss+=x[i]*x[i];double inv=1.0/std::sqrt(ss+1e-6);for(int i=0;i<K;i++)x[i]*=inv*sc;};
  std::vector<double> q(H*K),k(H*K),v(H*K);
  for(int h=0;h<H;h++)for(int i=0;i<K;i++){q[h*K+i]=qc[h*K+i];k[h*K+i]=qc[2048+h*K+i];v[h*K+i]=qc[4096+h*K+i];}
  for(int h=0;h<H;h++){l2(&q[h*K],1.0/std::sqrt((double)K));l2(&k[h*K],1.0);}
  mx->rst=1;mx->start=0;mx->we_gsp=mx->we_gex=mx->we_gsg=mx->we_s=mx->we_q=mx->we_k=mx->we_v=mx->we_nlut=mx->we_w=mx->we_z=0;T(mx);T(mx);mx->rst=0;
  for(int i=0;i<LUTN;i++){mx->g_addr=i;mx->we_gsp=1;mx->g_din=(uint32_t)llround((double)lsp[i]*65536.0);T(mx);mx->we_gsp=0;
    mx->we_gex=1;mx->g_din=(uint32_t)llround((double)lex[i]*1073741824.0);T(mx);mx->we_gex=0;
    mx->we_gsg=1;mx->g_din=(uint32_t)llround((double)lsg[i]*16777216.0);T(mx);mx->we_gsg=0;
    mx->we_nlut=1;mx->n_addr=i;mx->n_din=(uint32_t)llround((double)lsg[i]*65536.0);T(mx);mx->we_nlut=0;}
  for(int i=0;i<K;i++){mx->we_w=1;mx->addr_wz=i;mx->din_w=QI(nw[i],65536);T(mx);}mx->we_w=0;
  std::vector<double> og(H*K);
  for(int h=0;h<H;h++){mx->gm_a=QI(a[h],1024);mx->gm_dt=QI(dt[h],1024);mx->gm_A=QI(A[h],16384);mx->gm_b=QI(b[h],1024);
    for(int i=0;i<K;i++){mx->addr_qkv=i;mx->we_q=1;mx->din_q=QI(q[h*K+i],16777216);mx->we_k=1;mx->din_k=QI(k[h*K+i],16777216);mx->we_v=1;mx->din_v=QI(v[h*K+i],16777216);T(mx);}mx->we_q=mx->we_k=mx->we_v=0;
    for(int i=0;i<K*K;i++){mx->we_s=1;mx->addr_s=i;mx->din_s=QI(rec[(size_t)h*K*K+i],16777216);T(mx);}mx->we_s=0;
    for(int i=0;i<K;i++){mx->we_z=1;mx->addr_wz=i;mx->din_z=QI(z[h*K+i],4096);T(mx);}mx->we_z=0;
    mx->start=1;T(mx);mx->start=0;int g=0;while(!mx->done&&g++<200000)T(mx);
    for(int i=0;i<K;i++){mx->raddr_y=i;mx->eval();og[h*K+i]=(double)(int32_t)mx->rdata_y/65536.0;}}
  double am=0;for(double x:og)am=std::max(am,std::fabs(x));double so=(am>0?am/127.0:1.0);
  std::vector<int8_t> ogi8(IN2);for(int i=0;i<IN2;i++)ogi8[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(og[i]/so)));
  mo->rst=1;mo->load_en=0;mo->w_en=0;T(mo);T(mo);mo->rst=0;
  for(int i=0;i<IN2;i++){mo->load_en=1;mo->x_byte=ogi8[i];T(mo);}mo->load_en=0;
  std::vector<double> mixraw; proj(mo,wout.data(),swo.data(),so,HID,IN2,30,mixraw); // Q14
  std::vector<double> mix(HID); for(int i=0;i<HID;i++)mix[i]=mixraw[i]/16384.0;

  // ===== residual, post-norm, MLP, residual =====
  std::vector<double> h1(HID); for(int i=0;i<HID;i++)h1[i]=(double)h_in[i]+mix[i];
  double var=0;for(double x:h1)var+=x*x;var/=HID;double inv=1.0/std::sqrt(var+1e-6);
  std::vector<double> hn2(HID); for(int i=0;i<HID;i++)hn2[i]=h1[i]*inv*(1.0+(double)pln[i]);
  double hm=0;for(double x:hn2)hm=std::max(hm,std::fabs(x));double shn=(hm>0?hm/127.0:1.0);
  std::vector<int8_t> hn2i8(HID);for(int i=0;i<HID;i++)hn2i8[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(hn2[i]/shn)));
  // gate, up (Q12)
  mv->rst=1;mv->load_en=0;mv->w_en=0;T(mv);T(mv);mv->rst=0;
  for(int i=0;i<HID;i++){mv->load_en=1;mv->x_byte=hn2i8[i];T(mv);}mv->load_en=0;
  std::vector<double> gate,up; proj(mv,wg.data(),swg.data(),shn,INTER,HID,28,gate);
  proj(mv,wu.data(),swu.data(),shn,INTER,HID,28,up);
  // swiglu (Q12): silu(gate)*up
  sw->rst=1;sw->in_vld=0;sw->we_lut=0;T(sw);T(sw);sw->rst=0;
  for(int i=0;i<LUTN;i++){double xv=(i-512)/64.0,sgv=1.0/(1.0+std::exp(-xv)); // sigmoid over [-8,8], Q16
    sw->we_lut=1;sw->addr_lut=i;sw->din_lut=(uint32_t)llround(sgv*65536.0);T(sw);}sw->we_lut=0;
  std::vector<double> gg; int gi=0;
  for(int i=0;i<INTER;i++){sw->in_vld=1;sw->g=(int32_t)gate[i];sw->u=(int32_t)up[i];T(sw);if(sw->out_vld)gg.push_back((double)(int32_t)sw->y/4096.0);}
  sw->in_vld=0;T(sw);if(sw->out_vld)gg.push_back((double)(int32_t)sw->y/4096.0);
  double gm=0;for(double x:gg)gm=std::max(gm,std::fabs(x));double sg=(gm>0?gm/127.0:1.0);
  std::vector<int8_t> gi8(INTER);for(int i=0;i<INTER;i++)gi8[i]=(int8_t)std::max(-127.0,std::min(127.0,std::round(gg[i]/sg)));
  // down_proj (Q14)
  md->rst=1;md->load_en=0;md->w_en=0;T(md);T(md);md->rst=0;
  for(int i=0;i<INTER;i++){md->load_en=1;md->x_byte=gi8[i];T(md);}md->load_en=0;
  std::vector<double> dnraw; proj(md,wd.data(),swd.data(),sg,HID,INTER,30,dnraw); // Q14
  std::vector<double> out(HID); for(int i=0;i<HID;i++)out[i]=h1[i]+dnraw[i]/16384.0;

  double worst=0,refmax=0;
  for(int i=0;i<HID;i++){worst=std::max(worst,std::fabs(out[i]-(double)hout[i]));refmax=std::max(refmax,std::fabs((double)hout[i]));}
  std::printf("FULL DeltaNet layer (mixer -> residual -> post-norm -> MLP -> residual):\n");
  std::printf("  out worst |diff| %.3e  (max %.3e, rel %.2e)\n", worst, refmax, worst/refmax);
  bool ok=(worst/refmax<6e-2);
  std::printf("%s\n", ok?"PASS  full DeltaNet LAYER == golden, end to end":"FAIL");
  return ok?0:1;
}
