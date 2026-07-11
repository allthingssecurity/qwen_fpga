// DeltaNet gate math, per head, no floating-point unit. Given a, dt, A=exp(A_log)
// and b:
//   g    = -A * softplus(a + dt)      softplus from a lookup table
//   gexp = exp(g)                     exp from a lookup table, output Q30
//   beta = sigmoid(b)                 sigmoid from a lookup table, output Q24
// gexp (Q30) and beta (Q24) are exactly the formats deltanet_head expects, so
// this drops straight in front of the recurrence. Checked in Verilator vs golden.

module gate_math #(
    parameter int LUTN = 1024
) (
    input  logic               clk,
    input  logic               rst,
    // load the three tables
    input  logic               we_sp,   // softplus, Q16
    input  logic               we_ex,   // exp,      Q30
    input  logic               we_sg,   // sigmoid,  Q24
    input  logic [9:0]         addr_lut,
    input  logic [31:0]        din_lut,
    // per-head inputs
    input  logic               in_vld,
    input  logic signed [31:0] a,    // Q10
    input  logic signed [31:0] dt,   // Q10
    input  logic signed [31:0] A,    // Q14
    input  logic signed [31:0] b,    // Q10
    output logic [31:0]        gexp, // Q30
    output logic [31:0]        beta, // Q24
    output logic               out_vld
);
    logic [31:0] splut [0:LUTN-1];
    logic [31:0] exlut [0:LUTN-1];
    logic [31:0] sglut [0:LUTN-1];

    function automatic logic [9:0] clamp_idx (input logic signed [31:0] v);
        if (v < 0)            return 10'd0;
        else if (v > LUTN-1)  return LUTN-1;
        else                  return v[9:0];
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin out_vld <= 0; gexp <= 0; beta <= 0; end
        else begin
            if (we_sp) splut[addr_lut] <= din_lut;
            if (we_ex) exlut[addr_lut] <= din_lut;
            if (we_sg) sglut[addr_lut] <= din_lut;

            out_vld <= in_vld;
            if (in_vld) begin
                // each table read is linearly interpolated between adjacent
                // entries, so accuracy is not limited by the bin width
                logic signed [31:0] adt, off, g;
                logic [9:0]  i0;
                logic [4:0]  fr5;
                logic [3:0]  fr4;
                logic signed [63:0] lo, hi, sp, gprod;

                // softplus(a+dt): range [-16,16], 32 units/bin
                adt = a + dt;
                off = adt + (16 <<< 10);
                i0  = clamp_idx(off >>> 5);  fr5 = off[4:0];
                lo  = $signed({32'd0, splut[i0]});
                hi  = $signed({32'd0, splut[(i0==LUTN-1)?i0:i0+1]});
                sp  = lo + (((hi - lo) * $signed({59'd0, fr5})) >>> 5);   // Q16

                gprod = $signed(A) * sp;                 // Q30
                g = -(gprod >>> 20);                     // Q10, g <= 0

                // exp(g): range [-16,0], 16 units/bin. g <= 0, so at the top of
                // the range exp(0) = 1 exactly, which the table does not store.
                off = g + (16 <<< 10);
                if ((off >>> 4) >= LUTN) begin
                    gexp <= (32'd1 <<< 30);                     // exp(0) = 1, Q30
                end else begin
                    i0  = clamp_idx(off >>> 4);  fr4 = off[3:0];
                    lo  = $signed({32'd0, exlut[i0]});
                    // top bin interpolates up to exp(0) = 1 (Q30)
                    hi  = (i0 == LUTN-1) ? (64'sd1 <<< 30) : $signed({32'd0, exlut[i0+1]});
                    gexp <= (lo + (((hi - lo) * $signed({60'd0, fr4})) >>> 4));  // Q30
                end

                // sigmoid(b): range [-16,16], 32 units/bin
                off = b + (16 <<< 10);
                i0  = clamp_idx(off >>> 5);  fr5 = off[4:0];
                lo  = $signed({32'd0, sglut[i0]});
                hi  = $signed({32'd0, sglut[(i0==LUTN-1)?i0:i0+1]});
                beta <= (lo + (((hi - lo) * $signed({59'd0, fr5})) >>> 5));  // Q24
            end
        end
    end
endmodule
