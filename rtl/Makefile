# Simulate and check the Qwen RTL modules against the golden model with Verilator.
#
#   make -C rtl gemv     # build + run the int8 GEMV check
#
# Each target verilates one module plus its C++ testbench and runs it from the
# repo root so the artifacts/tv_*.bin paths resolve.

VFLAGS := --cc --exe --build -Wno-WIDTH -Wno-UNOPTFLAT -I.

.PHONY: gemv clean
gemv:
	cd .. && verilator $(VFLAGS) -Irtl rtl/gemv_i8.sv rtl/tb/sim_gemv.cpp \
		--top-module gemv_i8 -o sim_gemv >/dev/null 2>&1
	cd .. && ./obj_dir/sim_gemv

clean:
	rm -rf ../obj_dir
