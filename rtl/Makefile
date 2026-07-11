# Simulate and check the Qwen RTL modules against the golden model with Verilator.
#
#   make -C rtl gemv     # build + run the int8 GEMV check
#
# Each target verilates one module plus its C++ testbench and runs it from the
# repo root so the artifacts/tv_*.bin paths resolve.

VFLAGS := --cc --exe --build -Wno-WIDTH -Wno-UNOPTFLAT -I.

.PHONY: gemv matvec deltanet rmsnorm swiglu conv softmax rope gate l2norm gnorm mixer all clean
all: gemv matvec deltanet rmsnorm swiglu conv softmax rope gate l2norm gnorm mixer

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

clean:
	rm -rf ../obj_dir
