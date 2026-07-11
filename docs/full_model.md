# The whole model, one decode step through the RTL

`make -C rtl model` runs a full forward pass of Qwen3.5-0.8B, one decode step, on
the real Verilog blocks, and checks the predicted token against the fp32 reference.

## What it does

1. `scripts/export_model_tv.py` warms every layer's state with a short prompt using
   the golden model, snapshots the state, then dumps everything for one more token:
   all 24 layers' int8 weights and scales, the per-layer warm states (conv and
   recurrence for the DeltaNet layers, KV cache for the attention layers), the LUTs,
   the final norm, the tied embedding table, and the golden logits. Each layer record
   carries a magic marker so a misaligned read trips an assert instead of producing
   quiet garbage.
2. `rtl/tb/sim_model.cpp` is the harness. It plays the part the on-chip controller
   would: embedding lookup, then for each of the 24 layers in the real pattern
   ([DeltaNet x3, attention] x6) it dispatches to the DeltaNet or attention datapath,
   threads the hidden state from one layer into the next, and at the end applies the
   final norm and the tied output head to pick the next token.

Every matvec, the conv, the DeltaNet recurrence, the softmax, and the swiglu are the
real RTL modules, instantiated once and reused across all 24 layers. The light
elementwise steps (the norms, silu, l2 norm, residual adds, and the attention
scores and context dot products) are proven glue in the harness, the same split used
in the single-layer co-sims.

## Result

The model predicts the same next token as the fp32 reference. The hidden-state error
entering each layer stays flat near 2 percent across all 24 layers, so the match is
stable rather than luck. It runs in a few seconds.

## The precision finding

The first full run predicted the wrong token. The error was not int8 quantization: a
pure int8 model (per-row int8 weights, per-vector int8 activations, in fp32
arithmetic) predicts the correct token, so the int8 scheme is accurate enough. The
error was also not saturation (zero saturations counted) and not the DeltaNet
recurrence (its fixed-point core matches fp32 to under 0.2 percent).

It was the fixed-point format of one intermediate: the DeltaNet in_proj output was
kept in Q10 (an int16 with 10 fractional bits). The in_proj values span about plus
or minus 29, which fills the int16 range and leaves only 10 fractional bits, so small
channels get coarse absolute precision. The next step, the per-head l2 norm, turns
that coarse precision on a small-magnitude head into a large direction error, which
the recurrence then amplifies. Over 24 layers this flipped the top-1 prediction to a
near neighbor.

The fix is to carry that one path with more fractional bits. The matvec output width
is now a parameter; the in_proj uses a 32-bit output at Q20 (a lower dequant shift so
the multiplier still fits int32), and the conv, whose inputs are already 32-bit,
consumes it directly. With that, per-layer error drops from over 20 percent to about
2 percent and the prediction is correct. This is ordinary fixed-point range-and-
precision tuning, and it is the kind of thing only a full-depth run surfaces; the
single-layer checks passed at Q10 because they never accumulated.

## What this is not

This runs in simulation. No token has been generated on the FPGA. The forward pass
here drives the RTL blocks from a C++ harness that stands in for the controller and
streams weights from a file; on silicon that controller is an FSM and the weights
stream from HBM. That step is still ahead.
