# We Put a Piece of Qwen on an FPGA. Here Is What Actually Worked

There is a tempting version of this story where we say we ran Qwen on an FPGA and
leave it at that.

That would sound impressive. It would also be wrong.

What we actually did is more useful. We built a correct software reference for
Qwen3.5-0.8B, translated its important operations into hardware, ran the complete
24-layer model through those hardware blocks in simulation, found and fixed a
precision problem that only appeared at full depth, and then took the core int8
matrix-vector engine all the way to a real AWS F2 FPGA.

On the chip, that engine produced 64 correct results out of 64. The result matched
our golden reference exactly.

The full model is not on the FPGA yet. No prompt has gone into the chip and no token
has come out of it. The controller and the HBM weight-streaming path still have to
be built.

This distinction matters. In hardware work, a clean boundary between what is proven
and what is planned is not modesty. It is the work.

## Why try an FPGA in the first place?

A CPU executes a stream of instructions. A GPU has many fixed compute units and is
very good at doing the same operation over a large amount of data. An FPGA is
different. It contains programmable logic, registers, memory blocks, multipliers,
and a large routing fabric. We describe a circuit, and the chip becomes that
circuit.

This makes an FPGA a good place to explore what a specialised language-processing
chip could look like. We can remove the parts meant for general-purpose computing
and build only the datapath needed by the transformer.

For single-token LLM decoding, the main limit is often memory bandwidth. To produce
one token, the machine must read the model weights and perform a relatively small
amount of work per byte. The arithmetic matters, but keeping the weights moving
without stalls matters even more.

Qwen3.5-0.8B is an interesting model for this experiment. Its text path has 24
layers arranged as three Gated DeltaNet layers followed by one full-attention layer,
repeated six times. Only six layers maintain a growing KV cache. The other 18 use a
fixed recurrent state. That makes the model a sensible candidate for a streaming
design.

At int8, the model needs about 0.7586 GB of weight reads per generated token. The AWS
F2 device has 16 GiB of HBM and a peak memory bandwidth near 460 GB/s. In a separate
on-chip bandwidth test, we measured 426.14 GB/s of reads. The simple division gives
a theoretical ceiling near 562 tokens per second.

That is a projection, not a Qwen benchmark. Qwen has not produced tokens on this
FPGA. We use the number only to explain why the architecture is worth pursuing.

## We started with software, not Verilog

Before writing hardware, we needed an answer key.

We wrote a from-scratch NumPy implementation of Qwen3.5-0.8B decoding and checked it
step by step against the PyTorch model. The worst hidden-state deviation was
1.9e-06, and the worst logit deviation was 7.5e-06.

This reference model caught details that would have been painful to debug later. For
example, the model uses two RMSNorm conventions. One uses a gain of 1 plus the
stored weight because its parameters are centred at zero. The gated DeltaNet norm
uses the stored weight directly. Both look like RMSNorm in a diagram, but confusing
them makes the model quietly drift away from the correct output.

We then quantised the weights per output channel to int8 and tested the result over
multiple prompts. Int8 weights with floating-point activations had 97.9 percent
top-1 agreement. Int8 weights and int8 activations had 94.8 percent agreement. More
importantly for this build, the quantised reference still predicted the correct next
token used in our full-model verification.

Only after this did we begin treating the software output as golden data for the
hardware.

## Building the transformer one block at a time

The first hardware-shaped version was written in HLS C++. It helped us explore the
datapath and verify the model structure. But the F2 Vitis flow could not give us a
loadable FPGA image, so real silicon required the RTL and HDK route.

We wrote the model blocks in Verilog and SystemVerilog. The int8 matrix-vector unit,
the Gated DeltaNet recurrence, RMSNorm, SwiGLU, causal convolution, softmax, rotary
position handling, gate functions, L2 normalisation, and gated normalisation were
all checked against golden vectors.

The process was deliberately boring:

1. Build one block.
2. Compare it with the software reference.
3. Fix it until the numbers agree.
4. Compose it into the next larger unit.

The blocks became complete DeltaNet and attention mixers. The mixers became complete
decoder layers with normalisation, residual paths, and the MLP. Finally, the same
RTL compute blocks were reused across all 24 layers for one complete decode step.

The simulation began with an embedding, followed the true layer pattern, threaded
the hidden state and warm recurrent or KV state through the model, applied the final
normalisation and tied output head, and selected the next token.

It predicted the same token as the fp32 reference. The hidden-state error stayed
near 2 percent through all 24 layers, so the matching token was not a lucky result.

This was the first point where we could say that the full model arithmetic worked
end to end in the hardware datapath. It was still a simulator running on a laptop,
but it was exercising the real RTL blocks.

## The wrong token taught us more than the right one

The first full-model run did not pass. It predicted the wrong token.

This was useful because the smaller tests had all looked fine. There were no
saturations. The int8 reference predicted the correct token. The fixed-point
DeltaNet recurrence matched floating point to under 0.2 percent. Yet the full model
drifted to a nearby but incorrect answer.

The problem was one intermediate format.

The DeltaNet input projection was stored as a 16-bit Q10 value. Its range was large
enough, but smaller channels lost too much fractional precision. The following
per-head L2 normalisation turned that small absolute error into a larger direction
error. The recurrence then carried the error forward, layer after layer.

We widened that path to a 32-bit Q20 format. The per-layer error dropped from more
than 20 percent to about 2 percent, and the final prediction became correct.

This is why an isolated kernel result is not enough. A format that passes one layer
can fail after 24. Full-depth verification showed us where the model was genuinely
sensitive.

## Then came the real chip

For silicon, we started with the workhorse of the network: the int8 matrix-vector
engine. Every projection in every layer depends on this arithmetic.

We wrapped the engine as an AWS F2 custom-logic design with an AXI-Lite interface.
The host writes a 1024-element int8 activation vector, per-row dequantisation
multipliers, and a 64 by 1024 weight tile into the FPGA. It starts the engine, waits
for completion, and reads back 64 int16 outputs.

The first complete FPGA build routed, but it missed the fixed 250 MHz timing target
by 0.8 ns. The critical path ran through a wide multiply-accumulate carry chain. F2
does not allow us to solve this by lowering the main clock. We reduced the engine to
one MAC lane per cycle, shortened the path, and rebuilt it.

The next build met timing with 0.028 ns to spare.

Then the first image loaded, but the engine did not start. The status register stayed
at zero. Our AXI-Lite slave was accepting a write only when the address and data
arrived in the same cycle. The simulator happened to present them together. The real
bus master did not.

We changed the slave to latch the address and data independently. We also changed
the testbench to deliver them separately so that this bug cannot hide there again.

The corrected FPGA image was loaded on an f2.6xlarge:

```text
AFI:  afi-0b5c8b485bb2ff119
AGFI: agfi-01bd303a0be9536ce
Clock: 250 MHz
Post-route WNS: +0.028 ns
```

The host sent the input and weights over PCIe, started the engine, and read the
results back:

```text
FPGA qwen_matvec_engine: 64/64 rows correct
y[0]=316 golden=316, status=0x1
PASS
```

That was the moment the project crossed from a simulation result into a silicon
result. It was a small tile, and it was only one compute block, but it was the real
block, on the real device, returning the exact expected answer.

## How the chip actually runs the math

This is the part worth understanding in detail, because it is what "ran on the FPGA"
really means. There is no processor executing our code on the chip. There is a
circuit, and the host talks to it through memory.

**1. The image becomes the circuit.** An AFI is our placed-and-routed design turned
into a configuration bitstream. `fpga-load-local-image -S 0 -I agfi-...` streams that
bitstream into the FPGA and reprograms its fabric. After that command, the LUTs, the
flip-flops, the DSP multiplier, and the BRAM blocks on the die are physically wired
into our matrix-vector engine. The chip is no longer generic; it is our circuit until
we clear or replace it.

**2. The host reaches the circuit through a PCIe memory window.** The AWS shell
exposes our engine's control bus (an AXI-Lite interface called OCL) as a region of
PCIe memory, BAR0. From the host, `fpga_pci_poke(addr, value)` is a write into that
window and `fpga_pci_peek(addr)` is a read. Each poke/peek becomes one AXI-Lite
transaction that arrives at our circuit as an address and a data word. In other
words, the host and the FPGA share an address map, and reading or writing an address
on the host side pushes data straight into or out of our hardware.

**3. Inside the circuit, addresses select on-chip memories.** Our engine decodes the
incoming address:
- writes to `0x01_0000+` land in the activation BRAM (the 1024 int8 input values),
- writes to `0x02_0000+` land in the multiplier BRAM (the per-row dequant scales),
- writes to `0x10_0000+` land in the weight BRAM (the 64 by 1024 int8 tile),
- a write to `0x00` sets the start bit,
- reads of `0x04` return the status bits, and reads of `0x08_0000+` return results.

So loading the problem is just the host writing into three on-chip memories through
the PCIe window, one 32-bit word at a time.

**4. A hardware state machine runs the matmul.** Writing the start bit triggers a
finite-state machine clocked at 250 MHz. It streams the resident activation into the
GEMV, then streams the weight tile one int8 per cycle. Every cycle the DSP multiplies
a weight by the matching activation and accumulates. At the end of each 1024-long
row it has one 32-bit dot product; the dequant stage multiplies by that row's scale
and shifts to an int16, which is written into the result BRAM. Sixty-four rows later
the machine sets the done bit. No host involvement during this — it is pure hardware
for the whole computation.

**5. The host reads the answer back.** The host polls the status address until the
done bit is set (we saw `status=0x1`), then reads the 64 result addresses. Those are
the int16 numbers the circuit computed, and they matched the golden reference exactly.

That is the entire mechanism: reprogram the fabric into our circuit, share an address
map over PCIe, write the inputs into on-chip memory, pulse a start bit, let a 250 MHz
state machine do the int8 multiply-accumulate in dedicated logic, and read the
results out of on-chip memory. It is the same shape the full model will use, only
with a controller sequencing many of these operations and the weights arriving from
HBM instead of from host writes.

## What is done, and what is not

The cleanest summary is this:

The complete Qwen3.5-0.8B datapath is proven for one decode step in simulation. The
core int8 matrix-vector datapath is proven on real FPGA silicon. They have not yet
been joined into a complete on-chip model.

Four substantial pieces remain.

1. An on-chip controller must sequence all 24 layers, select the right mixer, manage
   recurrent and KV state, and thread the hidden vector through the network.
2. The model weights must be striped across the HBM channels and streamed through a
   proper AXI memory path with enough outstanding reads to keep the compute busy.
3. The embedding lookup and the 248,320-entry tied output head must be integrated.
4. The larger design must place, route, and close timing at the F2 fixed clock.

This is not a small matter of increasing a loop count. The controller and HBM path
are new hardware, and the memory system will decide the actual token rate. A larger
custom-logic build will also expose new timing and interface problems. Based on the
bring-up so far, we should expect a few.

Still, the project is no longer a slide or a throughput calculation. We have a
verified software reference, a full-model hardware simulation, a precision format
that survives all 24 layers, a working F2 build and load flow, measured HBM
bandwidth, and one central compute engine returning exact results from the chip.

That is a useful place to stand. The arithmetic is no longer the question. The next
question is whether we can keep it fed, control it correctly, and make the whole
model live on the FPGA.

## Evidence in the repository

The claims above can be checked directly:

1. `golden/` contains the NumPy and int8 reference implementations.
2. `rtl/` contains the verified hardware blocks and model simulations.
3. `docs/full_model.md` describes the full 24-layer run and the Q10 to Q20 precision fix.
4. `hdk/README.md` documents the build, timing closure, FPGA image, and silicon flow.
5. `artifacts/fpga_run.txt` is the actual on-chip transcript with the AFI, load result,
   and 64 out of 64 comparison.

No prompt has run on the FPGA yet. When it does, the proof should look just as plain:
the image that ran, the input that went in, the token that came out, and the measured
time on the chip.
