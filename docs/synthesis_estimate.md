# Synthesis feasibility — ANALYTICAL estimate (not a Vitis report)

**Every number here is hand-computed from the kernel structure and the VU47P
datasheet. None of it comes from Vitis HLS or Vivado — there is no Vitis in this
environment.** Treat it as a go/no-go sanity check before spending the ~$50-150
and several hours a real build costs, not as ground truth. The real report
replaces this the moment `v++ -c` runs (see build_afi.md).

## VU47P budget (raw silicon, before the F2 shell)

| resource | VU47P | F2 shell (est. ~20%) | ~CL available |
|---|---|---|---|
| LUT | 1,303,680 | ~260k | ~1,040k |
| DSP48E2 | 9,024 | ~200 | ~8,800 |
| URAM (288Kb) | 960 (33.75 MB) | small | ~940 (33 MB) |
| BRAM (36Kb) | ~2,016 (8.86 MB) | ~400 | ~1,600 |

Shell overhead is an estimate; the aws-fpga F2 shell report gives the real
figure.

## URAM — the DeltaNet state. Comfortable fit.

DeltaNet recurrent state is the one large on-chip structure:

```
rec   18 layers x 16 heads x 128 x 128 x 4 B = 18.87 MB
conv  18 layers x 6144 x 4 x 4 B             =  1.77 MB
                                       total ~ 20.6 MB  =  61% of 33 MB URAM
```

Fits with headroom. This is the payoff of the hybrid architecture: the state is
context-independent, so URAM usage is fixed — it never grows with sequence
length. A dense model's KV cache would spill to HBM and grow unbounded.

## DSP — this forces a decision I deferred. Read carefully.

Compute must keep pace with bandwidth: at 424 GB/s of int8 weights, that's
**~424 G MAC/s**, i.e. 424e9 / 250e6 = **~1,700 MACs per cycle** at 250 MHz.

The csim GEMV does `float(int8_weight) * activation_fp32` — an **fp32** multiply.
An fp32 MAC costs ~5 DSP48E2 (FMUL ~3 + FADD ~2):

```
fp32-activation path:  1,700 MAC/cyc x 5 DSP = ~8,500 DSP  =  ~97% of CL DSP
```

That's the whole chip. Technically it fits, but with no margin and poor timing
odds. **The fix is standard and I had deferred it: quantise activations to int8
too.** Then the MAC is int8 x int8 -> int32, and DSP48E2 packs 2 such MACs each:

```
int8-activation path:  1,700 MAC/cyc / 2 per DSP = ~850 DSP  =  ~10% of CL DSP
```

Comfortable. **So the real design wants int8 activations, not just int8
weights.** In `golden/eval_quant.py` I kept activations fp32 and said they don't
matter for *bandwidth* — true, but they dominate the *DSP* budget.

**Measured** (`scripts/eval_act_int8.py`, per-vector symmetric int8 activations):

| datapath | top-1 vs fp32 | mean KL | DSP |
|---|---|---|---|
| W8A16 (int8 wt, fp act) | 97.9% | 1.2e-3 | ~8,500 (97%) |
| **W8A8 (int8 wt + act)** | **94.8%** | 1.2e-2 | **~850 (10%)** |

So int8 activations are the difference between a design that barely fits and one
with 9x DSP headroom — at a cost of ~3 points of top-1 agreement and ~10x KL.
That's real but modest, and the crude per-VECTOR scaling here is the floor:
per-group activation scales (e.g. groups of 128) would recover most of the gap.
**Verdict: W8A8 is the design point**, with per-group activation quant as the
tuning knob if 94.8% isn't enough. The open item from the first draft is now
closed with data.

## BRAM — fits, with the streaming buffers as the driver

- outstanding-read FIFOs: 384 deep x 64 B x (up to 16 masters) ≈ 384 KB
- weight double-buffer + activation ping-pong: a few hundred KB
- softmax / RoPE scratch: small

Order ~200-400 BRAM of ~1,600. Fits. (Estimate — the linker report is exact.)

## LUT / FF — the least certain number

16 HBM AXI masters + address generation + the GEMV reduction trees + delta-rule
datapath + softmax + RoPE (sin/cos via CORDIC or a LUT table) + control.
Rough order **300k-500k LUT (30-48% of CL)**. This is the estimate I have least
confidence in; only synthesis resolves it. If it overflows, the lever is fewer
GEMV lanes (we have DSP headroom to trade) or a smaller outstanding depth.

## Timing — the actual engineering risk

The bandwidth roofline assumes 250 MHz (clk_main_a0). The hazard is the fp32
reduction in every GEMV: `acc += w*x` has ~4-cycle fp-add latency, so a naive
inner loop is recurrence-limited to II≈4, not II=1. Standard fix: split the
reduction into N independent partial accumulators (or a log-depth adder tree)
and combine at the end — turns the II=1 pipeline real. This is known, bounded
HLS work, but it *is* work, and it's the thing most likely to need iteration to
hit timing on a near-full part. Budget the 3-5 build passes accordingly.

## Verdict

| axis | estimate | fits VU47P? |
|---|---|---|
| URAM (state) | 20.6 MB / 33 MB | **yes, 61%** |
| DSP (int8 act) | ~850 / 8,800 | **yes, 10%** |
| DSP (fp32 act) | ~8,500 / 8,800 | risky, 97% |
| BRAM | ~300 / 1,600 | yes |
| LUT | ~300-500k / 1,040k | probably, least certain |
| timing 250 MHz | needs reduction restructure | achievable, is work |

**Analytically it fits, comfortably, with int8 activations (W8A8, measured
94.8% top-1).** The two things synthesis will actually stress are LUT usage and
the int8-reduction timing — both have known levers. Nothing here is a
showstopper, but nothing here has been through a tool either.
