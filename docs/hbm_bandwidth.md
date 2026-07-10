# HBM bandwidth validation (the bottleneck test)

csim proved the arithmetic. This proves the **memory system** — the only thing
that decides whether we hit 515 tok/s or 88. The whole performance story is
sustaining bandwidth on the 758 MB/token weight stream; nothing else matters
for batch-1 decode.

## Why not FireSim

FireSim was the first instinct (cycle-accurate, FPGA-accelerated). But its
memory model, **FASED, models DDR3 — not HBM** — is capped at 4 channels, and
even on a real-HBM board the target sees FASED's DDR timing by design. Our
entire thesis is HBM's 16-channel, 460 GB/s behaviour, which is exactly what
FASED abstracts away. So we validate the memory system with **DRAMsim3**, which
has a real HBM2 device model (pseudo-channels, row-buffer, refresh), driven by
the *actual* packed weight layout.

## Method

- **Device model** (`sim/F2_HBM2_16ch.ini`): 16 channels x 1 GiB = 16 GiB,
  tCK tuned so aggregate peak = 460 GB/s. Matches F2's VU47P on **both**
  capacity and bandwidth. JEDEC HBM2 timings from DRAMsim3's stock config.
- **Access pattern** (`artifacts/access.txt`): the real 758 MB/token stream —
  every int8 payload, scale, and fp32 tensor, walked in kernel read order from
  the packed manifest. Not synthetic.
- **Harness** (`sim/hbm_bw.cpp`): issues 64 B read transactions into DRAMsim3,
  holds up to `depth` outstanding, ticks until drained, measures sustained
  GB/s -> tok/s. Outstanding depth is swept — it's a real RTL knob (AXI read
  IDs / prefetch FIFO depth).

## Result

Two levers decide everything, and both are design choices we now know to make:

| layout | depth 16 | depth 64 | depth 256 | depth 384+ |
|---|---|---|---|---|
| **sequential** (`rorabgbachco`) | 35 | 88 | 275 | — |
| **channel-interleaved** (`rorabgbacoch`) | 61 | 221 | 486 | **559** |

(tokens/sec; peak = 460 GB/s / 758.6 MB per token = 607 tok/s absolute ceiling)

- **Naive design → 88 tok/s.** Sequential layout at a shallow queue serialises
  16 channels into one at a time (the address map keeps a stream in one open row
  for ~1 KiB before striping). A from-scratch build lands here and *looks like
  failure* — 5.8-14.6% of peak.
- **Correct design → 559 tok/s, 92.1% of peak.** Channel-interleave the weight
  layout and run ~384 outstanding reads. Plateaus at 424 GB/s; deeper queues
  don't help (controller queue limit = 16 ch x 32 = 512).
- **96.8% row-buffer hit rate** (387,796 / 400,539 read commands). Kernel-order
  layout is near-ideal for DRAM; the 7.9% gap to peak is refresh + ACT/PRE
  overhead, not the access pattern.

559 tok/s exceeds the original 515 roofline estimate — that assumed 85%
utilisation; the measured pattern achieves 92%.

## Two design requirements this surfaced

Both would otherwise have cost real synthesis iterations (hours + $) to discover:

1. **Channel-interleave the packed weights.** `pack_weights.py` currently lays
   tensors out contiguously. The streamer (or the pack step) must stripe
   cache-line-granularity across HBM pseudo-channels, i.e. the deployed design
   needs the equivalent of `rorabgbacoch` mapping — channel bits low. Without
   it: ~2x slower.
2. **Deep outstanding-read queue (~384).** The weight stream is fully
   prefetchable (addresses known in advance), so this is achievable — but the
   AXI master / read-request FIFO must sustain hundreds of in-flight reads.
   256 x 64 B = 16 KiB of in-flight data. A shallow queue caps you at <50% peak.

## Honest caveats

- This models the HBM **device**, not the AWS F2 **shell's** AXI exposure (port
  count, width, clock, crossbar, and the noted XDMA limitation). The shell
  derates this device-level ceiling by an unknown factor — pin that down before
  quoting 559 as a system number.
- Read-only stream (correct: weights dominate; activations + 6 KB/token KV +
  18 MB DeltaNet state are on-chip). Zero write/turnaround traffic.
- Assumes compute never stalls the read stream — justified: DeltaNet has ~6x
  headroom (see `hls/`) and int8 GEMV MACs are far under the DSP budget.

## Reproduce

```bash
# build DRAMsim3 (one-time)
cmake -S third_party/DRAMsim3 -B third_party/DRAMsim3/build \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release
make -C third_party/DRAMsim3/build dramsim3

python3 scripts/emit_access.py        # real access stream from the manifest
make -C sim                           # build harness
make -C sim run                       # sweep depth on the F2 config

# the winning config:
./sim/build/hbm_bw sim/F2_HBM2_16ch_interleave.ini artifacts/access.txt 384
```
