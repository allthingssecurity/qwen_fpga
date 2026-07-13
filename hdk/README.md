# HDK / RTL path to silicon

This is the real hardware path: the verified RTL, wrapped as an AWS F2 custom logic
(CL), built to a DCP, ingested into an AFI, and loaded on an f2 card. It is the flow
that can actually make an FPGA image, unlike the Vitis flow (which only does hw_emu
on F2).

## What goes on the chip first

`design/qwen_matvec_engine.sv` is the first CL: the model-verified int8 matvec
datapath (`matvec_i8` = `gemv_i8` + `dequant`) wrapped in an AXI4-Lite slave so the
host drives it directly. The host writes the activation, per-row dequant multipliers,
and a weight tile into on-chip BRAM, pulses start, polls done, and reads the int16
results. `make -C rtl engine` checks it over its own AXI bus against the golden
dequant matvec (64/64 rows). This is the datapath every Qwen layer is built from,
put on real silicon. The DDR/HBM weight-streaming controller for the full 24-layer
model is the next CL on top of this.


## Timing note (real bring-up finding)

F2 fixes `clk_main_a0` at 250 MHz (4 ns) and does not allow lowering it, unlike F1.
The first build routed but missed timing by 0.8 ns: the worst path was the LANES=4
int8 multiply-accumulate, a 17-level chain with 10 CARRY8 blocks off a BRAM read.
Fix: LANES=1 (one int8 MAC per cycle), which roughly halves the carry chain and
closes 250 MHz. The weight BRAM is larger at LANES=1, so it moved to its own
address region (addr[20]). Throughput drops but this is a functional proof; the
streaming controller will pipeline the MAC to get the width back.

## The build, step by step (what was actually run)

On an FPGA Developer AMI instance (Vivado 2025.2, >=64 GiB RAM):

```bash
# 1. HDK
git clone --depth 1 -b f2 https://github.com/aws/aws-fpga.git
source /opt/Xilinx/2025.2/Vivado/settings64.sh   # Vivado first...
cd aws-fpga && source hdk_setup.sh                # ...then HDK (it checks for Vivado)

# 2. make a CL from the template and drop the engine in
cd hdk/cl/examples && cp -r CL_TEMPLATE CL_QWEN
mv CL_QWEN/design/CL_TEMPLATE.sv CL_QWEN/design/CL_QWEN.sv
cp <repo>/hdk/design/qwen_matvec_engine.sv <repo>/rtl/{matvec_i8,gemv_i8,dequant}.sv CL_QWEN/design/
#   rename module CL_TEMPLATE -> CL_QWEN, defines CL_NAME -> CL_QWEN,
#   rename build/scripts/synth_CL_TEMPLATE.tcl -> synth_CL_QWEN.tcl,
#   replace the OCL tie-off block with the qwen_matvec_engine instance (see cl/CL_QWEN.sv).
#   encrypt.tcl auto-globs design/*.sv, so no source list to edit.

# 3. sanity: out-of-context synth of just the engine for the VU47P (~5 min)
vivado -mode batch -source ooc.tcl   # synth_design -top qwen_matvec_engine
                                     # -part xcvu47p-fsvh2892-2-e -mode out_of_context
#   -> 0 errors, 0 critical warnings; DSPs for the MACs, BRAM/LUTRAM for the buffers.

# 4. full CL build -> DCP  (synth first, then place & route; hours)
export CL_DIR=$(pwd)/CL_QWEN
cd CL_QWEN/build/scripts
python3 aws_build_dcp_from_cl.py -c CL_QWEN
#   -> build/checkpoints/to_aws/*.Developer_CL.tar

# 5. ingest into an AFI
aws s3 cp build/checkpoints/to_aws/*.tar s3://<bucket>/dcp/
aws ec2 create-fpga-image \
    --input-storage-location Bucket=<bucket>,Key=dcp/<tar> \
    --logs-storage-location  Bucket=<bucket>,Key=logs/
#   -> FpgaImageId (afi-...) and FpgaImageGlobalId (agfi-...); wait for State=available.

# 6. load on an f2 and run the host
#    sudo fpga-load-local-image -S 0 -I <agfi-...>
#    then host peek/poke to the OCL BAR: load x/mult/weights, start, read y.
```

Anything that has not run to completion is called out in the top-level README status.
The OCL AXI-Lite map the engine exposes is in `design/qwen_matvec_engine.sv`.
