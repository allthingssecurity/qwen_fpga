# Simulate and check the Qwen RTL modules against the golden model with Verilator.
#
#   make -C rtl gemv     # build + run the int8 GEMV check
#
# Each target verilates one module plus its C++ testbench and runs it from the
# repo root so the artifacts/tv_*.bin paths resolve.

VROOT := $(shell verilator --getenv VERILATOR_ROOT)

VFLAGS := --cc --exe --build -Wno-WIDTH -Wno-UNOPTFLAT -I.

.PHONY: gemv matvec deltanet rmsnorm swiglu conv softmax rope gate l2norm gnorm mixer mixer16 front mixerfull attnfull layer attnlayer model all clean
all: gemv matvec deltanet rmsnorm swiglu conv softmax rope gate l2norm gnorm mixer mixer16 front mixerfull attnfull layer attnlayer model

gemv:
	cd .. && verilator $(VFLAGS) -Irtl rtl/gemv_i8.sv rtl/tb/sim_gemv.cpp \
		--top-module gemv_i8 -o sim_gemv >/dev/null 2>&1
	cd .. && ./obj_dir/sim_gemv

matvec:
	cd .. && verilator $(VFLAGS) -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv \
		rtl/tb/sim_matvec.cpp --top-module matvec_i8 -o sim_matvec >/dev/null 2>&1
	cd .. && ./obj_dir/sim_matvec

deltanet:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/deltanet_head.sv \
		rtl/tb/sim_deltanet.cpp --top-module deltanet_head -o sim_deltanet >/dev/null 2>&1
	cd .. && ./obj_dir/sim_deltanet

rmsnorm:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/rmsnorm.sv \
		rtl/tb/sim_rmsnorm.cpp --top-module rmsnorm -o sim_rmsnorm >/dev/null 2>&1
	cd .. && ./obj_dir/sim_rmsnorm

swiglu:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/swiglu.sv \
		rtl/tb/sim_swiglu.cpp --top-module swiglu -o sim_swiglu >/dev/null 2>&1
	cd .. && ./obj_dir/sim_swiglu

conv:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/conv1d_tap.sv \
		rtl/tb/sim_conv.cpp --top-module conv1d_tap -o sim_conv >/dev/null 2>&1
	cd .. && ./obj_dir/sim_conv

softmax:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/softmax.sv \
		rtl/tb/sim_softmax.cpp --top-module softmax -o sim_softmax >/dev/null 2>&1
	cd .. && ./obj_dir/sim_softmax

rope:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/rope.sv \
		rtl/tb/sim_rope.cpp --top-module rope -o sim_rope >/dev/null 2>&1
	cd .. && ./obj_dir/sim_rope

gate:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/gate_math.sv \
		rtl/tb/sim_gate.cpp --top-module gate_math -o sim_gate >/dev/null 2>&1
	cd .. && ./obj_dir/sim_gate

l2norm:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/l2norm.sv \
		rtl/tb/sim_l2norm.cpp --top-module l2norm -o sim_l2norm >/dev/null 2>&1
	cd .. && ./obj_dir/sim_l2norm

gnorm:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/gated_norm.sv \
		rtl/tb/sim_gnorm.cpp --top-module gated_norm -o sim_gnorm >/dev/null 2>&1
	cd .. && ./obj_dir/sim_gnorm

mixer:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/deltanet_mixer_core.sv \
		rtl/gate_math.sv rtl/deltanet_head.sv rtl/gated_norm.sv \
		rtl/tb/sim_mixer.cpp --top-module deltanet_mixer_core -o sim_mixer >/dev/null 2>&1
	cd .. && ./obj_dir/sim_mixer

mixer16:
	cd .. && verilator $(VFLAGS) -Wno-BLKSEQ -Irtl rtl/deltanet_mixer_core.sv \
		rtl/gate_math.sv rtl/deltanet_head.sv rtl/gated_norm.sv \
		rtl/tb/sim_mixer16.cpp --top-module deltanet_mixer_core -o sim_mixer16 >/dev/null 2>&1
	cd .. && ./obj_dir/sim_mixer16

# multi-module co-sim (matvec + conv), linked by hand
front:
	cd .. && rm -rf obj_dir
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmv --Mdir obj_dir --top-module matvec_i8 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/conv1d_tap.sv --prefix Vcv --Mdir obj_dir --top-module conv1d_tap >/dev/null 2>&1
	cd .. && make -C obj_dir -f Vmv.mk >/dev/null 2>&1 && make -C obj_dir -f Vcv.mk >/dev/null 2>&1
	cd .. && clang++ -O2 -std=c++17 -Iobj_dir -I$(VROOT)/include -I$(VROOT)/include/vltstd rtl/tb/sim_front.cpp $(VROOT)/include/verilated.cpp $(VROOT)/include/verilated_threads.cpp obj_dir/Vmv__ALL.a obj_dir/Vcv__ALL.a -pthread -o obj_dir/sim_front >/dev/null 2>&1
	cd .. && ./obj_dir/sim_front

# full mixer co-sim: 4 real RTL modules linked by hand
mixerfull:
	cd .. && rm -rf obj_dir
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmv --Mdir obj_dir --top-module matvec_i8 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmo --Mdir obj_dir --top-module matvec_i8 -GIN=2048 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/conv1d_tap.sv --prefix Vcv --Mdir obj_dir --top-module conv1d_tap >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/deltanet_mixer_core.sv rtl/gate_math.sv rtl/deltanet_head.sv rtl/gated_norm.sv --prefix Vmix --Mdir obj_dir --top-module deltanet_mixer_core >/dev/null 2>&1
	cd .. && for m in Vmv Vmo Vcv Vmix; do make -C obj_dir -f $$m.mk >/dev/null 2>&1; done
	cd .. && clang++ -O2 -std=c++17 -Iobj_dir -I$(VROOT)/include -I$(VROOT)/include/vltstd rtl/tb/sim_mixerfull.cpp $(VROOT)/include/verilated.cpp $(VROOT)/include/verilated_threads.cpp obj_dir/Vmv__ALL.a obj_dir/Vmo__ALL.a obj_dir/Vcv__ALL.a obj_dir/Vmix__ALL.a -pthread -o obj_dir/sim_mixerfull >/dev/null 2>&1
	cd .. && ./obj_dir/sim_mixerfull

# synthesizable AXI-Lite matvec engine (first-silicon datapath) co-sim
engine:
	cd .. && rm -rf hdk/obj_dir
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-WIDTHTRUNC -Irtl hdk/design/qwen_matvec_engine.sv rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vqwen_matvec_engine --top-module qwen_matvec_engine --Mdir hdk/obj_dir >/dev/null 2>&1
	cd .. && make -C hdk/obj_dir -f Vqwen_matvec_engine.mk >/dev/null 2>&1
	cd .. && clang++ -O2 -std=c++17 -Ihdk/obj_dir -I$(VROOT)/include -I$(VROOT)/include/vltstd hdk/verif/sim_engine.cpp $(VROOT)/include/verilated.cpp $(VROOT)/include/verilated_threads.cpp hdk/obj_dir/Vqwen_matvec_engine__ALL.a -pthread -o hdk/obj_dir/sim_engine >/dev/null 2>&1
	cd .. && ./hdk/obj_dir/sim_engine

# whole model, one decode step through all 24 layers on the real RTL blocks
model:
	cd .. && python3 scripts/export_model_tv.py
	cd .. && rm -rf obj_dir
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmv --Mdir obj_dir --top-module matvec_i8 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmvw --Mdir obj_dir --top-module matvec_i8 -GOUTW=32 -GSHIFT=6 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmo --Mdir obj_dir --top-module matvec_i8 -GIN=2048 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmd --Mdir obj_dir --top-module matvec_i8 -GIN=3584 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/conv1d_tap.sv --prefix Vcv --Mdir obj_dir --top-module conv1d_tap >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/deltanet_mixer_core.sv rtl/gate_math.sv rtl/deltanet_head.sv rtl/gated_norm.sv --prefix Vmix --Mdir obj_dir --top-module deltanet_mixer_core >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/softmax.sv --prefix Vsm --Mdir obj_dir --top-module softmax >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/swiglu.sv --prefix Vsw --Mdir obj_dir --top-module swiglu >/dev/null 2>&1
	cd .. && for m in Vmv Vmvw Vmo Vmd Vcv Vmix Vsm Vsw; do make -C obj_dir -f $$m.mk >/dev/null 2>&1; done
	cd .. && clang++ -O2 -std=c++17 -Iobj_dir -I$(VROOT)/include -I$(VROOT)/include/vltstd rtl/tb/sim_model.cpp $(VROOT)/include/verilated.cpp $(VROOT)/include/verilated_threads.cpp obj_dir/Vmv__ALL.a obj_dir/Vmvw__ALL.a obj_dir/Vmo__ALL.a obj_dir/Vmd__ALL.a obj_dir/Vcv__ALL.a obj_dir/Vmix__ALL.a obj_dir/Vsm__ALL.a obj_dir/Vsw__ALL.a -pthread -o obj_dir/sim_model >/dev/null 2>&1
	cd .. && ./obj_dir/sim_model

# full attention decoder layer co-sim (attention mixer + residual + post-norm + MLP + residual)
attnlayer:
	cd .. && rm -rf obj_dir
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmv --Mdir obj_dir --top-module matvec_i8 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmo --Mdir obj_dir --top-module matvec_i8 -GIN=2048 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmd --Mdir obj_dir --top-module matvec_i8 -GIN=3584 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/softmax.sv --prefix Vsm --Mdir obj_dir --top-module softmax >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/swiglu.sv --prefix Vsw --Mdir obj_dir --top-module swiglu >/dev/null 2>&1
	cd .. && for m in Vmv Vmo Vmd Vsm Vsw; do make -C obj_dir -f $$m.mk >/dev/null 2>&1; done
	cd .. && clang++ -O2 -std=c++17 -Iobj_dir -I$(VROOT)/include -I$(VROOT)/include/vltstd rtl/tb/sim_attnlayer.cpp $(VROOT)/include/verilated.cpp $(VROOT)/include/verilated_threads.cpp obj_dir/Vmv__ALL.a obj_dir/Vmo__ALL.a obj_dir/Vmd__ALL.a obj_dir/Vsm__ALL.a obj_dir/Vsw__ALL.a -pthread -o obj_dir/sim_attnlayer >/dev/null 2>&1
	cd .. && ./obj_dir/sim_attnlayer

# full DeltaNet decoder layer co-sim (mixer + residual + post-norm + MLP + residual)
layer:
	cd .. && rm -rf obj_dir
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmv --Mdir obj_dir --top-module matvec_i8 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmo --Mdir obj_dir --top-module matvec_i8 -GIN=2048 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmd --Mdir obj_dir --top-module matvec_i8 -GIN=3584 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/conv1d_tap.sv --prefix Vcv --Mdir obj_dir --top-module conv1d_tap >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/deltanet_mixer_core.sv rtl/gate_math.sv rtl/deltanet_head.sv rtl/gated_norm.sv --prefix Vmix --Mdir obj_dir --top-module deltanet_mixer_core >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/swiglu.sv --prefix Vsw --Mdir obj_dir --top-module swiglu >/dev/null 2>&1
	cd .. && for m in Vmv Vmo Vmd Vcv Vmix Vsw; do make -C obj_dir -f $$m.mk >/dev/null 2>&1; done
	cd .. && clang++ -O2 -std=c++17 -Iobj_dir -I$(VROOT)/include -I$(VROOT)/include/vltstd rtl/tb/sim_layer.cpp $(VROOT)/include/verilated.cpp $(VROOT)/include/verilated_threads.cpp obj_dir/Vmv__ALL.a obj_dir/Vmo__ALL.a obj_dir/Vmd__ALL.a obj_dir/Vcv__ALL.a obj_dir/Vmix__ALL.a obj_dir/Vsw__ALL.a -pthread -o obj_dir/sim_layer >/dev/null 2>&1
	cd .. && ./obj_dir/sim_layer

# full attention mixer co-sim
attnfull:
	cd .. && rm -rf obj_dir
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmv --Mdir obj_dir --top-module matvec_i8 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/matvec_i8.sv rtl/gemv_i8.sv rtl/dequant.sv --prefix Vmo --Mdir obj_dir --top-module matvec_i8 -GIN=2048 >/dev/null 2>&1
	cd .. && verilator --cc -Wno-WIDTH -Wno-UNOPTFLAT -Wno-BLKSEQ -Irtl rtl/softmax.sv --prefix Vsm --Mdir obj_dir --top-module softmax >/dev/null 2>&1
	cd .. && for m in Vmv Vmo Vsm; do make -C obj_dir -f $$m.mk >/dev/null 2>&1; done
	cd .. && clang++ -O2 -std=c++17 -Iobj_dir -I$(VROOT)/include -I$(VROOT)/include/vltstd rtl/tb/sim_attnfull.cpp $(VROOT)/include/verilated.cpp $(VROOT)/include/verilated_threads.cpp obj_dir/Vmv__ALL.a obj_dir/Vmo__ALL.a obj_dir/Vsm__ALL.a -pthread -o obj_dir/sim_attnfull >/dev/null 2>&1
	cd .. && ./obj_dir/sim_attnfull

clean:
	rm -rf ../obj_dir
