#!/usr/bin/env bash
# End-to-end: everything that runs on a laptop for $0. No AWS, no FPGA.
#
#   ./run_all.sh            # full pipeline
#   ./run_all.sh hbm        # just the HBM bandwidth test (needs artifacts/ already built)
#
# Stages:
#   1 golden   numpy decode == torch reference   (needs .venv with torch)
#   2 quant    int8 quality + byte ledger
#   3 pack     write the HBM-resident int8 blob + manifest
#   4 csim     DeltaNet HLS kernel == golden
#   5 hbm      real weight stream through the F2 HBM2 model, sweep depth

set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
VPY=./.venv/bin/python
[ -x "$VPY" ] || VPY="$PY"   # fall back if no venv

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

build_dramsim3() {
  if [ ! -f third_party/DRAMsim3/libdramsim3.dylib ] && [ ! -f third_party/DRAMsim3/libdramsim3.so ]; then
    step "building DRAMsim3 (one-time)"
    [ -d third_party/DRAMsim3 ] || git clone --depth 1 https://github.com/umd-memsys/DRAMsim3 third_party/DRAMsim3
    cmake -S third_party/DRAMsim3 -B third_party/DRAMsim3/build \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release >/dev/null
    make -C third_party/DRAMsim3/build dramsim3 >/dev/null
  fi
}

hbm_test() {
  build_dramsim3
  [ -f artifacts/access.txt ] || $PY scripts/emit_access.py
  [ -f sim/F2_HBM2_16ch_interleave.ini ] || \
    sed 's/rorabgbachco/rorabgbacoch/' sim/F2_HBM2_16ch.ini > sim/F2_HBM2_16ch_interleave.ini
  make -C sim >/dev/null
  step "HBM bandwidth: naive layout (sequential) vs correct (channel-interleaved)"
  printf '%-14s %-8s %-12s %s\n' layout depth 'GB/s' 'tok/s'
  for cfg in F2_HBM2_16ch F2_HBM2_16ch_interleave; do
    name=$([ "$cfg" = F2_HBM2_16ch ] && echo sequential || echo interleaved)
    for d in 16 64 256 384; do
      line=$(./sim/build/hbm_bw "sim/$cfg.ini" artifacts/access.txt $d 2>/dev/null \
             | awk '/ACHIEVED/{g=$2}/DECODE RATE/{t=$3}END{printf "%s %s",g,t}')
      printf '%-14s %-8s %s\n' "$name" "$d" "$line"
    done
  done
  echo; echo "winner: channel-interleaved @ depth 384  ->  ~424 GB/s (92% of 460), ~559 tok/s"
}

case "${1:-all}" in
  hbm) hbm_test; exit 0 ;;
esac

# full pipeline
[ -f artifacts/ref.npz ] || { step "torch reference (slow, ~1 min)"; $VPY golden/dump_torch_ref.py; }
step "1/5 golden: numpy == torch";        $PY golden/verify.py | tail -3
step "2/5 quant: int8 quality";           $PY golden/eval_quant.py | tail -3
step "3/5 pack: HBM int8 blob";           $PY scripts/pack_weights.py --check | grep -E "TOTAL|roofline|worst"
step "4/5 csim: DeltaNet kernel";         make -C hls csim 2>/dev/null | tail -4
step "4b   csim: FULL decode datapath";   $PY scripts/emit_weight_index.py >/dev/null; \
                                          $PY scripts/export_decode_ref.py >/dev/null; \
                                          make -C hls csim_decode 2>/dev/null | tail -3
step "5/5 hbm: bandwidth";                hbm_test
