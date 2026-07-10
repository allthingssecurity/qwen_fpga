# Silicon validation on AWS F2

Note up front: Qwen never ran on the FPGA in this work. What ran on the chip was
a memory bandwidth benchmark. The model itself has not been built into hardware.
This document records proving the FPGA delivery path and measuring HBM bandwidth,
not running the model.

This is the record of getting a design onto a real F2 FPGA end to end, and what
it took to get there. The goal of the whole project is to run Qwen3.5-0.8B decode
on the chip. Before writing the full model in hardware, we proved every step of
the delivery path on real silicon with small designs, so that a large amount of
model RTL is never blocked by a broken build, image, or load step discovered at
the end.

## The path we could not use, and why

The first attempt used the Vitis HLS flow. The Qwen decode kernels are written
and verified as HLS C++, so this route would have run the exact code we already
proved correct. A 32 channel HBM bandwidth kernel was written, compiled, and
taken all the way through place and route on the real VU47P part. Real numbers
came out of it: about 39,400 LUTs (3.2 percent), 272 BRAM (14 percent), 4 DSP,
0 URAM, closing timing at 188.5 MHz after auto frequency scaling.

The build produced a valid xclbin, but the next step failed. On F2, AWS does not
currently support turning a Vitis xclbin into an Amazon FPGA Image. The aws-fpga
f2 branch states this directly: the Vitis flow currently supports hardware
emulation only, and hardware builds and AFI creation are not supported at this
time. So the Vitis route cannot reach silicon on F2 right now.

## The path that works: HDK / RTL

The supported route to an AFI on F2 is the HDK flow, where the custom logic is
Verilog wrapped in the AWS shell, built to a design checkpoint, and submitted to
`create-fpga-image`. To de-risk this before committing to Qwen RTL, we validated
the whole chain with the smallest available example.

### Step 1: build a tiny design to a checkpoint

Example: `hdk/cl/examples/cl_demo/cl_axil_reg_access`, a minimal register access
CL. Built with the HDK Python flow:

```
aws_build_dcp_from_cl.py -c cl_axil_reg_access
```

Result: clean build in about 25 minutes, 0 failed nets, 0 errors, a
`Developer_CL.tar` checkpoint tarball.

### Step 2: checkpoint to AFI

This is the step the Vitis route could not do. Upload the tarball to S3 and call
`create-fpga-image`:

```
aws ec2 create-fpga-image \
  --name qwen-smoke-axil \
  --input-storage-location Bucket=<bucket>,Key=dcp/smoke_cl.tar \
  --logs-storage-location  Bucket=<bucket>,Key=logs/
```

Result: it accepted the checkpoint and returned an image id and a global id. The
image ingested to state `available` in under an hour. AFI creation on F2 works
through the HDK flow.

### Step 3: load the image on a real f2 and talk to it

On an f2.6xlarge, install the runtime tools and load the image:

```
source sdk_setup.sh
sudo fpga-load-local-image -S 0 -I agfi-...
sudo fpga-describe-local-image -S 0 -R
```

Result: the image reported `loaded ok`, and the FPGA device id changed to show
the custom logic was live. The example host program then wrote values to the CL
registers and read results back:

```
PASS: 49174301 + 3972913407 = 4022087708 (carry = 0)
...
PASS all operations. Number of carry positive operations = 496
TEST PASSED
```

The chip received data from the host, computed with it, and returned correct
results. Host to FPGA communication over the shell works.

## What this proves

Every link in the delivery chain now works on real F2 silicon:

1. HDK build to a design checkpoint
2. Checkpoint to an AFI with `create-fpga-image`
3. AFI loaded onto an f2 with `fpga-load-local-image`
4. Host driving the custom logic and reading correct results back

None of these steps is a dead end. The bandwidth benchmark `cl_mem_perf` is the
next design through the same chain. It runs 32 HBM channels at 450 MHz and
reports achieved read bandwidth, which is the ceiling on Qwen decode speed, and
it doubles as the proven HBM wrapper and clocking template that the Qwen custom
logic will be built on.

## Measured HBM bandwidth

`cl_mem_perf` was built through the same chain, closed timing at 450 MHz on all
32 HBM channels, ingested to an AFI, loaded onto a real f2.6xlarge, and run:

```
WR Bandwidth = 433.69 GBytes/s
RD Bandwidth = 426.14 GBytes/s
RD Latency   = 268 ns
TEST PASSED
```

Read bandwidth is 92.6 percent of the 460 GB/s device peak. For Qwen decode,
which reads about 0.7586 GB of weights per token, this gives:

```
426.14 GB/s / 0.7586 GB = 562 tokens per second
```

This matches the DRAMsim3 model, which predicted 424 GB/s and 559 tokens per
second before any hardware was in hand. The measurement confirms the ceiling on
real silicon.

## What comes next

With the chain proven and a working HBM template in hand, the remaining work is
the Qwen decode datapath itself in RTL: weight streaming from HBM, the Gated
DeltaNet layers, the gated attention layers, and the MLP, wired into the same CL
that already reaches silicon. That is the actual model on chip, and it now sits
on a foundation where the build, the image, the load, and the host path are all
known to work.
