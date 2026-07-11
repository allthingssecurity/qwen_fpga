// SwiGLU elementwise: y = silu(g) * u, where silu(x) = x * sigmoid(x). The
// sigmoid is a lookup table (the standard way to do a nonlinearity in hardware),
// covering [-8, 8] in LUTN entries, values in Q16. Activations g, u, y are Q12.
//
// One element per cycle: index the table from g, read sigmoid, form silu = g*sig,
// then y = silu*u. Checked in Verilator against golden silu(g)*u.

module swiglu #(
    parameter int Q    = 12,     // activation fractional bits
    parameter int LUTN = 1024,   // table entries over [-8, 8]
    parameter int SHFT = 6       // = Q - log2(LUTN/16) ; maps g(Q12) -> index
) (
    input  logic               clk,
    input  logic               rst,
    // load the sigmoid table (Q16)
    input  logic               we_lut,
    input  logic [9:0]         addr_lut,
    input  logic [31:0]        din_lut,
    // streaming elementwise
    input  logic               in_vld,
    input  logic signed [31:0] g,
    input  logic signed [31:0] u,
    output logic signed [31:0] y,
    output logic               out_vld
);
    logic [31:0] slut [0:LUTN-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            out_vld <= 0; y <= 0;
        end else begin
            if (we_lut) slut[addr_lut] <= din_lut;

            out_vld <= in_vld;
            if (in_vld) begin
                logic signed [31:0] idx_s;
                logic [9:0]         idx;
                logic signed [63:0] sig, silu, yv;
                idx_s = (g >>> SHFT) + (LUTN/2);          // (g + 8*2^Q) mapped to [0,LUTN)
                if (idx_s < 0)            idx = 0;
                else if (idx_s > LUTN-1)  idx = LUTN-1;
                else                      idx = idx_s[9:0];
                sig  = $signed({32'd0, slut[idx]});        // Q16, non-negative
                silu = ($signed(g) * sig) >>> 16;          // Q12
                yv   = (silu * $signed(u)) >>> Q;          // Q12
                y <= yv[31:0];
            end
        end
    end
endmodule
