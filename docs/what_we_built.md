# Qwen on FPGA: what we built, how, and what "full model" actually takes

Written to be read start to finish. It starts from FPGA basics, explains why an
FPGA is a natural fit for language-model inference, walks the whole path we took,
states plainly what runs on real silicon today, and is honest about the gap to
running the full model. No marketing.

---

## 1. What an FPGA actually is

A CPU is fixed silicon that reads instructions and executes them one stream at a
time. A GPU is fixed silicon with thousands of small cores doing the same operation
across lots of data. Both are built once at the factory and you write software for
them.

An **FPGA (Field-Programmable Gate Array) is a chip whose logic is not fixed.** It
is a sea of tiny hardware primitives plus a routing fabric, and you decide how they
connect. You are not writing software that runs on a processor; you are describing a
**circuit**, and the chip becomes that circuit. The primitives on the AWS F2 chip
(a Xilinx Virtex UltraScale+ VU47P) are:

- **LUTs (lookup tables):** the atoms of logic. A LUT is a tiny truth table that can
  become any small boolean function (an AND, a mux, part of an adder). There are
  ~1.3 million of them.
- **FFs (flip-flops):** 1-bit registers. They hold a value and update on the clock
  edge. This is how a circuit has state and pipelining.
- **DSP blocks:** hardened multiply-accumulate units. ~9,000 of them. Each does a
  fast multiply plus add. This is what you build matrix math out of.
- **BRAM / URAM:** on-chip memory blocks, small but very fast, right next to the
  logic. Tens of megabits total. This is your scratchpad.
- **HBM:** 16 GiB of high-bandwidth DRAM stacked on the same package, ~460 GB/s.
  This is where big things (like model weights) live because they don't fit on-chip.

You describe the circuit in an HDL (Hardware Description Language) — we used
**Verilog/SystemVerilog**. A toolchain (Xilinx Vivado) then does:

1. **Synthesis** — turn your Verilog into a netlist of LUTs/FFs/DSPs/BRAM.
2. **Place** — decide which physical LUT/DSP/BRAM on the die each piece uses.
3. **Route** — connect them through the fabric.
4. **Timing closure** — prove every signal arrives before the next clock edge.

That last step is the one people underestimate, and it bit us (section 6).

### Why "everything happens at once" matters

On a CPU, `y = W @ x` is a loop: millions of multiply-add instructions, one after
another. On an FPGA you can lay down a datapath where, every single clock cycle, a
row of multiply-accumulates happens **in parallel in dedicated hardware**, and the
weights stream past that hardware. There is no instruction fetch, no cache miss, no
scheduler. The circuit does exactly one job, and it does it every cycle. That is the
whole appeal for a workload as repetitive as a transformer.

---

## 2. Why an FPGA is close to a "Language Processing Unit"

A GPU is a general parallel machine that happens to be good at LLMs. The idea of a
**Language Processing Unit (LPU)** is the opposite: a chip whose datapath *is* the
transformer, with nothing wasted on generality. FPGAs let you prototype exactly that
without taping out a custom chip. Two facts about LLM **decoding** (generating one
token at a time) make this fit especially well:

**Decode is memory-bandwidth-bound, not compute-bound.** To produce one token you
read every weight of the model exactly once and do relatively little math per weight.
So the speed limit is *how fast you can stream the weights past the compute*, not how
many multipliers you have. A 0.8B-parameter model at int8 is ~0.76 GB per token; at
460 GB/s that ceiling is ~600 tokens/sec. The design that wins is the one that keeps
the weights flowing and never stalls — a **streaming datapath**, which is precisely
what an FPGA is good at.

**int8 is enough, and int8 is cheap in hardware.** Full-precision floats are
expensive in fabric. 8-bit integer multiply-accumulate is tiny and fast. If you
quantize the model to int8 (we did, and verified it keeps the right answer), each
weight is one byte and each multiply is trivial. This is why the automated
"compile PyTorch straight to FPGA" projects blow up — they keep everything in wide
floats on-chip and need ~140x more logic than exists. The frugal path is:
hand-designed int8, weights in HBM, streaming past a small compute datapath. That is
the design philosophy of this whole repo, and it is what an LPU-style chip would do.

So: an FPGA running a hand-built int8 streaming transformer datapath is a working
model of what a dedicated language-processing chip would be. That is the thesis.

---

## 3. What we actually built (the whole path)

The work went from a correct software reference down to real silicon, in layers,
each one checked against the one above it. Nothing was claimed done until its check
passed.

**(a) A golden software model.** A from-scratch numpy implementation of Qwen3.5-0.8B
decoding, checked bit-for-bit against PyTorch. This is the reference every later
stage is measured against. It captures the real architecture: 24 layers in a
`[DeltaNet x3, attention] x6` pattern, two very different mixer types, RMSNorm, the
gated DeltaNet recurrence, gated attention with QK-norm and partial RoPE, and the
SwiGLU MLP. We also built the int8-quantized version and confirmed it still predicts
the same tokens.

**(b) The datapath in HLS C++.** A first hardware-shaped description (High-Level
Synthesis) of the decode datapath, checked against the golden model. Useful for
exploring structure, but on F2 the HLS/Vitis flow can only do emulation — it cannot
produce a loadable FPGA image. That pushed us to the RTL/HDK flow for real silicon.

**(c) Every block written in real RTL (Verilog) and verified.** One module at a time,
each simulated in Verilator against the golden vectors, exact for integer blocks and
within tight fixed-point tolerance otherwise:
- `gemv_i8` / `matvec_i8` — the int8 matrix-vector engine (the workhorse; every
  projection in every layer is this).
- `deltanet_head` / `deltanet_mixer_core` — the gated DeltaNet recurrence in Q24
  fixed point (the hardest block; no floating point unit anywhere).
- `rmsnorm`, `swiglu`, `conv1d_tap`, `softmax`, `rope`, `gate_math`, `l2norm`,
  `gated_norm` — norms, activations, the causal conv, the attention softmax, all in
  integer/fixed-point math with lookup tables for the nonlinearities.

**(d) The blocks composed into whole layers, end to end.** Both mixer types
assembled and checked against golden (`make -C rtl mixerfull`, `attnfull`), then both
complete decoder layers — norm, mixer, residual, norm, MLP, residual
(`make -C rtl layer`, `attnlayer`), each matching the reference to ~2%.

**(e) The whole 24-layer model, in simulation, predicting the right token.**
`make -C rtl model` runs one full decode step through all 24 layers on the real RTL
blocks — embedding in, the layers sequenced in the true pattern, hidden state
threaded through, final norm, output head — and it predicts the **same next token as
the fp32 reference**. This runs in a Verilog simulator on a laptop, not on the chip,
but it proves the whole model's hardware datapath is correct.

**(f) The core compute block taken all the way to real silicon.** We wrapped the int8
matvec engine as an AWS F2 custom-logic design driven by the host over an AXI-Lite
bus, built it through Vivado to a placed-and-routed image that **closes timing at
250 MHz**, ingested it into a real **AFI (Amazon FPGA Image)**, loaded it onto a real
f2 FPGA, and ran it. The host wrote an int8 input and a weight tile into the chip,
said go, and read back results that matched the golden math **64 out of 64, exactly**.
Transcript and AFI id are in `artifacts/fpga_run.txt`.

---

## 4. What runs on real silicon today — stated plainly

The one thing running on the actual chip is the **int8 matrix-multiply datapath**,
on a small test tile (a 1024-long input times a 64x1024 weight tile), driven by the
host, producing correct results. That is the exact arithmetic block every layer of
Qwen is built from.

What is **not** on the chip: the full 24-layer model, the controller that sequences
it, weight streaming from HBM, the embedding, and the output head. No prompt has gone
to the chip; no token has come out of the chip. "Prompt in, text out" still runs on
the CPU/simulation model.

So the accurate one-liner is: **the compute unit is proven on silicon; the full
model is proven in simulation; the two have not been joined yet.**

---

## 5. Findings along the way (this is where the real work was)

These are the non-obvious problems that only showed up because each stage was
actually checked. They are worth listing because they are what "engineering" meant
here.

- **RMSNorm convention.** The model has two different RMSNorm variants (one uses a
  `1+weight` gain with zero-centered parameters, one uses plain weight). Getting this
  wrong made the golden model silently diverge. Fixed by matching the real weights.
- **Fixed-point overflow in the recurrence.** 32x32 multiplies in the DeltaNet
  recurrence overflowed when computed in 32 bits before the shift; sign-extending to
  64 bits first fixed a 6000x error.
- **Precision, not quantization, was the accuracy limit at full depth.** The first
  full-model sim predicted the *wrong* token. It was not int8 (a pure int8 model
  predicts correctly) and not the recurrence (exact to 0.2%). It was one intermediate
  format: the DeltaNet input projection kept only 10 fractional bits, and after the
  per-head L2 norm that coarseness became a large direction error that the recurrence
  amplified over 24 layers. Widening that one path to a 32-bit/Q20 format dropped the
  per-layer error from >20% to ~2% and the prediction became correct. This is the
  kind of thing only a full-depth run surfaces.
- **F2 fixes the clock at 250 MHz.** The first silicon build routed but **missed
  timing by 0.8 ns**. Unlike the older F1, F2 does not let you slow the fabric clock.
  The critical path was the multiply-accumulate carry chain. Narrowing it to one MAC
  per cycle closed timing (+0.021 ns). Real, and only visible after place-and-route.
- **AXI handshake bug that only appeared on hardware.** The first AFI loaded but the
  engine never started (status read back 0). The cause: my bus slave required the
  address and data to arrive on the *same* cycle; a real bus master presents them
  separately. The simulator happened to drive them together, so it passed in sim and
  failed on silicon. Fixed by latching address and data independently — and the
  testbench now drives them apart so this can't hide again.

Every one of these is a real hardware-bring-up lesson, not a detail.

---

## 6. Is running full Qwen "just a matter of scaling"? Honestly, no.

This is the important part, and I want to be straight rather than encouraging.

**What genuinely carries over (a lot):** the entire compute story is done and proven.
The int8 matvec runs on silicon. Every other block (recurrence, norms, softmax,
swiglu, conv, RoPE, gates) is written and verified in simulation. The full model's
math is proven correct end to end in the simulator. The AFI build/load/run flow works.
The quantization scheme is validated. None of that has to be reinvented. In that
sense the hard *arithmetic* and the *toolchain path* are behind us.

**What is genuinely new work (not scaling):** getting from "a matmul runs on the chip"
to "a prompt produces a token on the chip" needs four things that do not exist as
hardware yet, and they are real hardware, not bigger versions of what's there:

1. **A controller.** Today the thing that sequences the 24 layers, threads the hidden
   state from one into the next, and orchestrates embedding and the output head is a
   C++ program on the host (the simulation harness). On the chip that has to become a
   **finite-state machine in Verilog** that drives the datapath. That is a new module,
   and it is the brain of the design.
2. **HBM weight streaming.** The silicon demo put a tiny weight tile in on-chip BRAM
   and loaded it with slow host writes. The real model's 0.8B weights do not fit
   on-chip at all — they must **stream from HBM** through an AXI memory interface,
   continuously, without stalling the compute. Building that streaming path (and
   keeping it fed) is the single biggest new piece, and it is exactly where the
   performance lives.
3. **Embedding lookup and the output head.** Turning a token id into a vector at the
   input, and turning the final vector into a next-token choice over a 248k-word
   vocabulary at the output. Both are memory-bound lookups/matmuls that need to be
   wired into the datapath.
4. **A much larger CL build.** All of the above is a substantially bigger custom-logic
   design than the one-block engine we shipped. It will take more resources, will be
   harder to close at 250 MHz, and will take several build-and-fix iterations — the
   same kind we already hit, but more of them.

So the honest framing is not "we have a small kernel, now just make it big." It is:
**we proved the arithmetic and the full-model math, and we proved one block on real
silicon; the remaining work is building the control and memory-streaming hardware
that turns that block into the whole model on the chip.** That is a real project — a
meaningful fraction of the total effort still ahead — but it is now standing on top
of a verified datapath and a working silicon flow, which is the part most people
never get through.

---

## 7. Where to see each claim in the repo

- `golden/` — the numpy reference, checked against PyTorch.
- `rtl/` + `make -C rtl <block>` — every verified hardware block.
- `make -C rtl model` — the whole model in simulation, predicting the golden token.
- `make -C rtl engine` — the silicon engine verified over its own AXI bus.
- `hdk/` + `hdk/README.md` — the F2 build flow, the CL, the host program.
- `artifacts/fpga_run.txt` — the actual on-silicon run: AFI id, load, 64/64 result.
- `docs/full_model.md` — the full-model simulation and the precision finding.

The top-level `README.md` "Status" section is kept honest: what runs on the chip,
what runs only in simulation, and what has not happened.
