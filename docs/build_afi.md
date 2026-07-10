# Building the AFI and running on F2 — the AWS-side runbook

Everything up to this point ran on a laptop for $0. This is the part that needs
**Vitis (Linux-only, licensed) and an AWS F2 instance** — neither of which
exists in the dev environment where the rest was built. These are the exact
commands to take the verified kernel to silicon; they have not been executed
here, so treat this as the tested-shaped recipe, not a transcript.

## 0. Prereqs

- AWS account with an **F instance quota** (F2 is not enabled by default —
  open a support case to raise the F on-demand quota).
- An S3 bucket in your build region for the AFI ingestion.
- Regenerate the artifacts the build consumes:
  ```bash
  python3 scripts/pack_weights.py            # artifacts/qwen35_int8.bin
  python3 scripts/gen_offsets.py             # hls/src/weight_offsets.hpp
  ```

## 1. Launch a build instance (NOT an F2 — synthesis is CPU-only)

FPGA Developer AMI (free; Vitis/Vivado 2025.2 licence included, AWS-parts-only).
Use a compute/memory instance with **≥64 GiB RAM** — a near-full VU47P P&R will
blow past 32 GiB.

```bash
# z1d.2xlarge (high clock, 64 GiB) or r6i.4xlarge (128 GiB headroom)
# from the FPGA Developer AMI, then:
git clone https://github.com/aws/aws-fpga.git -b f2
cd aws-fpga && source vitis_setup.sh
export AWS_PLATFORM=$AWS_PLATFORM_XILINX_F2   # set by vitis_setup on F2 AMIs
```

## 2. Software emulation first — free, catches interface bugs (no FPGA)

`hw_emu` runs the synthesised RTL cycle-accurately with no FPGA. This is the
first place the Vitis interface + the kernel pragmas are actually exercised.

```bash
cd hls/vitis
v++ -c -t hw_emu --platform $AWS_PLATFORM --config qwen.cfg \
    -k qwen_decode_kernel -I../src qwen_kernel.cpp -o qwen.hw_emu.xo
v++ -l -t hw_emu --platform $AWS_PLATFORM --config qwen.cfg \
    qwen.hw_emu.xo -o qwen.hw_emu.xclbin
XCL_EMULATION_MODE=hw_emu ./qwen_host qwen.hw_emu.xclbin artifacts/qwen35_int8.bin
```

The host (`hls/vitis/qwen_host.cpp`, written) loads `qwen35_int8.bin` to HBM,
feeds the prompt tokens, and reproduces the argmax stream from `tb_decode`.

**Scope hw_emu to ONE layer, not full generation.** hw_emu is cycle-accurate
RTL cosim — kHz-class throughput. One full 24-layer token is billions of
cycle-events and can take hours to simulate; a whole prompt is impractical. Emulate
a single kernel invocation (one layer, a couple of tokens) to validate the HBM
AXI interface, the m_axi burst behaviour, and the achieved initiation interval.
Full-model correctness is already settled in csim; hw_emu is here to confirm the
RTL and timing, not the math. Cap generation with a small `n_new` (1-2) and
expect it to be slow even so.

## 3. Hardware build -> xclbin (this is the hours-long, $ step)

```bash
v++ -c -t hw --platform $AWS_PLATFORM --config qwen.cfg \
    -k qwen_decode_kernel -I../src qwen_kernel.cpp -o qwen.xo
v++ -l -t hw --platform $AWS_PLATFORM --config qwen.cfg \
    qwen.xo -o qwen.xclbin        # <- Vivado synth + P&R here, ~6-16 h
```

The **HLS report from `v++ -c`** replaces docs/synthesis_estimate.md with real
LUT/DSP/URAM/BRAM/fmax. Check it *before* the long link step — if DSP blew up,
that's the int8-activation decision (see the estimate doc) to make first.

## 4. xclbin -> AFI

```bash
$AWS_FPGA_REPO_DIR/Vitis/tools/create_vitis_afi.sh \
    -xclbin=qwen.xclbin \
    -o=qwen \
    -s3_bucket=<your-bucket> -s3_dcp_key=qwen/dcp -s3_logs_key=qwen/logs
# emits qwen.awsxclbin + an afi-*.txt with the AFI/AGFI ids; ingestion ~1-3 h
aws ec2 describe-fpga-images --fpga-image-ids <afi-id> \
    --query 'FpgaImages[0].State.Code'      # poll until "available"
```

## 5. Run on a real F2

```bash
# launch f2.6xlarge (1 FPGA, 16 GiB HBM, ~$1.98/hr on-demand us-east-1)
# FPGA Runtime AMI, then:
./qwen_host qwen.awsxclbin artifacts/qwen35_int8.bin --prompt "Explain FPGAs."
```

This is the first execution on silicon, and the first **measured** tok/s — the
number that finally replaces the 559 tok/s DRAMsim3 projection with reality,
derated by whatever the F2 shell's AXI/XDMA path actually costs.

## Cost / time to first working AFI

| item | ~cost |
|---|---|
| build instance, 3-5 passes x ~12 h @ $0.68-1.01/hr | $25-60 |
| AFI ingestion | (undocumented; likely free) |
| f2.6xlarge validation, a few hours @ $1.98/hr | $5-15 |
| **total compute** | **~$50-150** |

Engineer time (timing closure iterations, the int8-activation accuracy check,
the XRT host) dominates the compute cost.

## What still needs writing before step 2 works

1. `hls/vitis/qwen_host.cpp` — XRT host (~150 lines).
2. The KV-cache-in-HBM substitution inside the kernel wrapper (csim uses a
   std::vector view; hardware needs `kv_hbm` + offsets). Math unchanged.
3. The int8-activation accuracy check (numpy, free) — decides the DSP budget.
4. The fp32-reduction restructuring for II=1 timing (HLS, in the GEMV).

These are the honest remaining gaps between "csim passes" and "runs on F2."

Status: (1) is written (`qwen_host.cpp`). (2)-(4) remain, and all four are best
done ON the build instance where Vitis can actually check them.
