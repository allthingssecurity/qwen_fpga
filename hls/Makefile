# csim: compile the HLS kernel with a host compiler and check it against the
# numpy golden vectors. No Vitis, no FPGA, no AWS. Run this before you ever pay
# for synthesis.
#
#   make -C hls csim
#
# `#pragma HLS` is invisible to clang, so the same source is the synthesis
# source. -ffp-contract=off keeps clang from fusing a*b+c into an FMA, which
# would change rounding and make the bit-comparison against numpy meaningless.

CXX      ?= clang++
# -Wno-unused-label: the loop labels are consumed by Vitis pragmas and reports,
# but look dead to clang.
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label \
            -ffp-contract=off

BUILD := ../artifacts
BIN   := $(BUILD)/tb_deltanet
BIN_DECODE := $(BUILD)/tb_decode
BIN_RUN := $(BUILD)/run_hls
HDRS := $(wildcard src/*.hpp)

.PHONY: csim csim_decode run_hls clean
# unit test: the DeltaNet kernel alone
csim: $(BIN)
	cd .. && ./artifacts/tb_deltanet

# end-to-end: the whole decode datapath vs golden int8
csim_decode: $(BIN_DECODE)
	cd .. && ./artifacts/tb_decode

$(BIN): tb/tb_deltanet.cpp src/deltanet.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tb/tb_deltanet.cpp -o $@

$(BIN_DECODE): tb/tb_decode.cpp $(HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tb/tb_decode.cpp -o $@

# generative prompt driver over the synthesisable top
run_hls: $(BIN_RUN)
$(BIN_RUN): tb/run_hls.cpp $(HDRS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tb/run_hls.cpp -o $@

clean:
	rm -f $(BIN) $(BIN_DECODE) $(BIN_RUN)
