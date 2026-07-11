// Dequantise a GEMV accumulator to a fixed-point activation, in synthesizable
// integer math. The fp32 dequant is acc * sx * sw[o]. Instead of a floating
// multiply, the host precomputes a fixed-point multiplier
//   mult[o] = round(sx * sw[o] * 2^(OUTQ+SHIFT))
// and the hardware does one integer multiply and an arithmetic right shift.
// The output y_q is the activation in Q(OUTQ) fixed point (int16). This is what
// real int8 inference does, no floating-point unit needed.
//
// Combinational here so the check lines up cycle-for-cycle with the GEMV; the
// synthesis version registers the multiply with matched pipelining.

module dequant #(
    parameter int SHIFT = 16,   // extra fractional bits in mult, removed by shift
    parameter int OUTQ  = 14    // fractional bits kept in the output (Q(OUTQ))
) (
    input  logic                in_vld,
    input  logic signed [31:0]  acc,
    input  logic signed [31:0]  mult,
    output logic signed [15:0]  y_q,
    output logic                out_vld
);
    logic signed [63:0] p;
    always_comb begin
        p = ($signed(acc) * $signed(mult)) + (64'sd1 <<< (SHIFT-1));  // round
        y_q = p >>> SHIFT;                                            // to Q(OUTQ)
        out_vld = in_vld;
    end
endmodule
