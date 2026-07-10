# HBM bandwidth harness. Links DRAMsim3 (built as a dylib under third_party).
#
#   make -C sim            # build
#   make -C sim run        # build + sweep outstanding depth
#
# DRAMsim3 must be built first:
#   cmake -S third_party/DRAMsim3 -B third_party/DRAMsim3/build \
#         -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release
#   make -C third_party/DRAMsim3/build dramsim3

D3   := ../third_party/DRAMsim3
BUILD := build
BIN  := $(BUILD)/hbm_bw

CXX      ?= clang++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -I$(D3)/src -I$(D3)/ext/headers
LDFLAGS  := -L$(D3) -ldramsim3 -Wl,-rpath,$(abspath $(D3))

.PHONY: all run clean
all: $(BIN)

$(BIN): hbm_bw.cpp
	@mkdir -p $(BUILD) ../artifacts/dramsim3_out
	$(CXX) $(CXXFLAGS) hbm_bw.cpp -o $@ $(LDFLAGS)

# sweep outstanding-request depth to find the bandwidth knee
run: $(BIN)
	@cd .. && for d in 8 16 32 64 128 256 512; do \
	  ./sim/build/hbm_bw sim/F2_HBM2_16ch.ini artifacts/access.txt $$d \
	    | grep -E "outstanding|ACHIEVED|DECODE"; echo; \
	done

clean:
	rm -f $(BIN)
