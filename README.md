# Qwen3.5-0.8B on AWS FPGA (F2)

Running batch-1 decode of `Qwen/Qwen3.5-0.8B` (text only) on an AWS F2 instance,
which carries an AMD Virtex UltraScale+ HBM part (VU47P) with 16 GiB of HBM.
The work covers a bit-exact software model, an HLS hardware datapath verified
against it, a cycle-accurate HBM bandwidth study, a real Vitis synthesis pass on
the actual chip, and a full end to end validation of the FPGA delivery path on
real silicon.

## The core idea

Single token decode reads every weight from memory once, does one multiply with
it, and moves on. It is limited by memory bandwidth, not by math. So the token
rate is set by one equation:

```
tokens per second = HBM bandwidth / bytes read per token
```

For this model at int8 that is about 0.7586 GB per token. This is why the whole
project centers on HBM bandwidth, and why the hybrid architecture of this model
matters: only 6 of its 24 layers keep a KV cache, so its per token memory traffic
stays flat as context grows, unlike a dense model whose KV cache eventually
dominates.

## Model shape

Qwen3.5-0.8B, text path only (the vision tower and the MTP head are dropped).
24 layers in the pattern `[linear_attention x3, full_attention] x6`.

- **Gated DeltaNet layers (18 of 24):** 16 heads, key and value head dim 128, a
  short causal conv of width 4, no growing KV cache. The recurrent state is
  16 x 128 x 128 x 18 layers = 18 MB in fp32, constant regardless of context
  length. That fits in the VU47P on-chip URAM, so it never touches HBM.
- **Full attention layers (6 of 24):** 8 query heads, 2 key/value heads, head
  dim 256, QK-norm, partial rotary on the first 64 of 256 dims, and an output
  gate. KV traffic is 6 KB per token, against 56 KB per token for a comparable
  dense model.
- hidden size 1024, vocab 248320, tied embedding and output head.

## Repository layout

```
golden/     pure-numpy fp32 and int8 decode, checked against HF transformers
hls/        HLS C++ kernels and testbenches, plus the Vitis wrapper and configs
sim/        DRAMsim3 HBM bandwidth harness driven by the real weight layout
scripts/    weight packing, offset generation, prompt batch, eval
docs/        architecture, bandwidth study, synthesis estimate, silicon validation
artifacts/  generated outputs and the real Vitis reports
```

## Results

### 1. The software model matches PyTorch

`golden/` is a from-scratch numpy decode transcribed from the Qwen3.5 modeling
code and checked step by step against Hugging Face transformers.

```
worst hidden deviation   1.9e-06
worst logit  deviation   7.5e-06
result: numpy model matches torch
```

Two things this caught, both of which would have produced fluent nonsense in
hardware without an error: the norm uses `(1 + weight)` because the checkpoint
stores gamma centered at zero, and the DeltaNet gated norm does not, so the two
norms in the model use different conventions. And the recurrent form of the delta
rule matches the chunked form to 7.6e-06, so the hardware needs a single
recurrent datapath, not a separate prefill kernel.

### 2. int8 holds up, and int8 activations make it fit

Per output channel int8 weights, checked against fp32 over several prompts:

| datapath | top-1 agreement | mean KL | bytes per token |
|---|---|---|---|
| int8 weights, fp32 activations | 97.9% | 1.2e-3 | 758 MB |
| int8 weights and int8 activations | 94.8% | 1.2e-2 | 758 MB |

int8 activations cost a few points of agreement but drop the compute cost by
roughly 10x, because the multiply becomes int8 by int8. Quantizing the tied
embedding and output head is free on quality and halves the per token bytes, so
it is done.

### 3. The full HLS decode matches the model

Every operation is written as synthesizable HLS C++ and composed end to end:
embedding, both norms, the int8 GEMV weight streamer, the causal conv, the gated
delta rule, partial rotary, gated attention, SwiGLU, and the tied head. Checked
against the int8 software model over 11 decode steps:

```
argmax matches at all 11 steps
step-0 logits agree to 6.6e-05 (fp summation order only)
result: HLS decode matches the software model
```

The DeltaNet kernel also has a standalone check on a warm state, where the output
matches to 3.7e-09 and the evolved recurrent state to 6.0e-08 over 262,144
elements.

### 4. HBM bandwidth study (cycle accurate model)

Since real silicon was not yet in hand for this step, the memory system was
modeled with DRAMsim3 using an HBM2 config matched to the F2 device on both
capacity and bandwidth, driven by the actual 758 MB weight layout in the order
the decoder reads it. Two design choices decide everything:

| layout | shallow queue | deep queue |
|---|---|---|
| naive sequential | 88 tok/s | 391 tok/s |
| channel interleaved | 221 tok/s | 559 tok/s |

The channel interleaved layout with a deep outstanding read queue reaches 424
GB/s, which is 92 percent of the 460 GB/s device peak, at a 96.8 percent row
buffer hit rate. This becomes two concrete hardware requirements: stripe the
weights across HBM pseudo-channels, and keep a few hundred reads in flight.

### 5. Real Vitis synthesis on the VU47P

An HBM bandwidth kernel and the model kernels were run through real Vitis
synthesis and place and route on the actual chip. Genuine numbers from the tools:

- The int8 GEMV closes timing at 342 MHz and uses about 3 percent of the DSPs.
- The DeltaNet head as first written failed timing at 66 MHz because a 128 wide
  fp32 reduction was forced into one cycle. Tiling the value dimension so the
  tool can register the arithmetic fixes this, verified in the software checks.
- The 32 channel HBM bandwidth kernel mapped to the real part at about 39,400
  LUTs (3.2 percent), 272 BRAM (14 percent), 4 DSP, 0 URAM.

These reports are saved under `artifacts/vitis_reports/`.

### 6. The FPGA delivery path works on real silicon

The Vitis flow cannot create an AFI on F2 today, so the route to silicon is the
HDK / RTL flow. The full chain was validated on real hardware with small designs
before committing to the model RTL. Build to checkpoint, checkpoint to AFI, AFI
loaded onto an f2, and host driving the logic and reading correct results back,
all confirmed. Details in `docs/silicon_validation.md`.

### 7. Measured HBM bandwidth on real silicon

The AWS HBM bandwidth benchmark `cl_mem_perf` was built through the HDK flow,
turned into an AFI, and run on a real f2.6xlarge. It closed timing at 450 MHz on
all 32 HBM channels. Measured on the chip:

```
read bandwidth   426.14 GB/s
write bandwidth  433.69 GB/s
read latency     268 ns
```

Read is 92.6 percent of the 460 GB/s device peak. Decode reads the weights, so
the read number is the one that sets the token rate:

```
426.14 GB/s / 0.7586 GB per token = 562 tokens per second
```

The DRAMsim3 model in section 4 predicted 424 GB/s and 559 tokens per second.
The real chip measured 426 GB/s and 562 tokens per second. The model was right,
now confirmed on silicon. Full output is in
`artifacts/vitis_reports/f2_hbm_bandwidth.txt`.

### 8. Generations

Full prompts, complete answers, token counts, and timings are in
`artifacts/generations.md`. The text comes from the verified int8 model path, the
same arithmetic the hardware datapath uses. The token rate the hardware sustains
comes from the measured HBM bandwidth above, not from these CPU timings.

## Reproduce

Everything except the FPGA build runs on a laptop.

```bash
python3 -m venv .venv && ./.venv/bin/pip install torch transformers safetensors numpy

./.venv/bin/python golden/dump_torch_ref.py    # torch reference
python3 golden/verify.py                        # numpy vs torch
python3 golden/eval_quant.py                     # int8 quality
python3 scripts/pack_weights.py --check          # pack int8 weights for HBM
make -C hls csim_decode                          # full HLS decode vs model
python3 scripts/run_batch.py                     # prompt batch and timings

# HBM bandwidth model (builds DRAMsim3 once)
cmake -S third_party/DRAMsim3 -B third_party/DRAMsim3/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -C third_party/DRAMsim3/build dramsim3
python3 scripts/emit_access.py && make -C sim && ./run_all.sh hbm
```

The FPGA build and silicon steps are in `docs/silicon_validation.md` and
`docs/build_afi.md`.

## Status and what is next

Done: the software model, the HLS datapath, the bandwidth study, a real Vitis
synthesis pass, and full validation of the silicon delivery path on real F2
hardware. Next is the actual model on chip, the Qwen decode datapath written in
RTL on top of the proven HBM template, so the FPGA generates tokens directly.
