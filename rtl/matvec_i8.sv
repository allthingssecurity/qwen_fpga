// A complete int8 matrix-vector primitive: the streaming int8 GEMV followed by
// the fixed-point dequant. Loads an int8 activation, streams int8 weights, and
// emits each output as a Q(OUTQ) fixed-point activation ready for the next op.
// This is the reusable building block every Qwen layer is made of.

module matvec_i8 #(
    parameter int IN    = 1024,
    parameter int OUT   = 512,
    parameter int LANES = 8,
    parameter int SHIFT = 16,   // dequant shift; lower it (with wider OUTW) for more frac bits
    parameter int OUTQ  = 14,
    parameter int OUTW  = 16    // output width; use 32 on the in_proj path (needs range+precision)
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    load_en,
    input  logic signed [7:0]       x_byte,
    input  logic                    w_en,
    input  logic [LANES*8-1:0]      w_word,
    input  logic signed [31:0]      mult,     // dequant multiplier for the current row
    output logic signed [OUTW-1:0]  y_q,
    output logic                    y_vld
);
    logic signed [31:0] acc;
    logic               acc_vld;

    gemv_i8 #(.IN(IN), .OUT(OUT), .LANES(LANES)) u_gemv (
        .clk(clk), .rst(rst),
        .load_en(load_en), .x_byte(x_byte),
        .w_en(w_en), .w_word(w_word),
        .acc_o(acc), .acc_vld(acc_vld)
    );

    dequant #(.SHIFT(SHIFT), .OUTQ(OUTQ), .OUTW(OUTW)) u_deq (
        .in_vld(acc_vld), .acc(acc), .mult(mult),
        .y_q(y_q), .out_vld(y_vld)
    );
endmodule
