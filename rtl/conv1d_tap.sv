// Depthwise causal conv1d, kernel 4, the short convolution at the front of each
// DeltaNet layer. Per channel the window is [state1, state2, state3, new] and the
// output is the dot product with that channel's 4 weights. The window shifting
// (state <- [state2,state3,new,...]) is trivial control handled by the datapath
// sequencer; this is the arithmetic core, one channel per cycle.
//
// Q10 fixed point (values reach ~20 here). Checked in Verilator against golden.

module conv1d_tap #(
    parameter int Q = 10
) (
    input  logic               clk,
    input  logic               rst,
    input  logic               in_vld,
    input  logic signed [31:0] w0, w1, w2, w3,   // window: state1,state2,state3,new
    input  logic signed [31:0] c0, c1, c2, c3,   // per-channel weights
    output logic signed [31:0] y,                // conv output (pre-silu)
    output logic               out_vld
);
    always_ff @(posedge clk) begin
        if (rst) begin y <= 0; out_vld <= 0; end
        else begin
            out_vld <= in_vld;
            if (in_vld) begin
                logic signed [63:0] acc;
                acc = $signed(w0) * $signed(c0)
                    + $signed(w1) * $signed(c1)
                    + $signed(w2) * $signed(c2)
                    + $signed(w3) * $signed(c3);   // Q20
                y <= acc >>> Q;                     // Q10
            end
        end
    end
endmodule
