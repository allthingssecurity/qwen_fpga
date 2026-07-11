// DeltaNet gated output norm: y = w * (o * rsqrt(mean(o^2)+eps)) * silu(z).
// The delta-rule output o can be tiny, so it comes in as Q24 (deltanet_head's
// format). rsqrt uses the integer isqrt+divide; silu uses an interpolated
// sigmoid table. w is Q16, z is Q12, y is Q16. No floating-point unit.
// Checked in Verilator against golden.

module gated_norm #(
    parameter int V    = 128,
    parameter int LGV  = 7,      // log2(V)
    parameter int LUTN = 1024
) (
    input  logic               clk,
    input  logic               rst,
    input  logic               we_lut,
    input  logic [9:0]         addr_lut,
    input  logic [31:0]        din_lut,     // sigmoid, Q16
    input  logic               we_o,
    input  logic [7:0]         addr_o,
    input  logic signed [31:0] din_o,       // Q24
    input  logic               we_w,
    input  logic [7:0]         addr_w,
    input  logic signed [31:0] din_w,       // Q16
    input  logic               we_z,
    input  logic [7:0]         addr_z,
    input  logic signed [31:0] din_z,       // Q12
    input  logic               start,
    output logic               done,
    input  logic [7:0]         raddr_y,
    output logic signed [31:0] rdata_y      // Q16
);
    logic [31:0]        slut [0:LUTN-1];
    logic signed [31:0] omem [0:V-1];
    logic signed [31:0] wmem [0:V-1];
    logic signed [31:0] zmem [0:V-1];
    logic signed [31:0] ymem [0:V-1];
    assign rdata_y = ymem[raddr_y];
    localparam logic [63:0] EPS_Q48 = 64'd281474977;   // round(1e-6 * 2^48)

    function automatic logic [63:0] isqrt (input logic [63:0] a);
        logic [63:0] rem, root, bitv;
        rem = a; root = 0; bitv = 64'h4000_0000_0000_0000;
        for (int s = 0; s < 32; s++) if (bitv > rem) bitv = bitv >> 2;
        for (int s = 0; s < 32; s++)
            if (bitv != 0) begin
                if (rem >= (root + bitv)) begin rem = rem - (root + bitv); root = (root >> 1) + bitv; end
                else root = root >> 1;
                bitv = bitv >> 2;
            end
        return root;
    endfunction

    typedef enum logic [2:0] {IDLE, ACC, RSQ, OUT, FIN} state_t;
    state_t st;
    logic [7:0] idx;
    logic signed [63:0] ss, inv;   // inv Q16

    always_ff @(posedge clk) begin
        if (rst) begin st <= IDLE; done <= 0; idx <= 0; ss <= 0; end
        else begin
            done <= 0;
            if (st == IDLE) begin
                if (we_lut) slut[addr_lut] <= din_lut;
                if (we_o) omem[addr_o] <= din_o;
                if (we_w) wmem[addr_w] <= din_w;
                if (we_z) zmem[addr_z] <= din_z;
            end
            case (st)
                IDLE: if (start) begin ss <= 0; idx <= 0; st <= ACC; end
                ACC: begin
                    logic signed [63:0] ov; ov = $signed(omem[idx]);
                    ss <= ss + (ov * ov);                           // Q48
                    if (idx == V-1) begin idx <= 0; st <= RSQ; end else idx <= idx + 1;
                end
                RSQ: begin
                    logic [63:0] r; logic signed [63:0] a;
                    a = (ss >>> LGV) + EPS_Q48;                     // mean + eps, Q48
                    r = isqrt(a << 16);                             // sqrt(a_real)*2^32
                    inv <= (64'sd1 <<< 48) / $signed(r);            // Q16
                    st <= OUT;
                end
                OUT: begin
                    logic signed [63:0] on, ow, sig, silu, yv;
                    logic signed [31:0] zoff; logic [9:0] i0; logic [6:0] fr;
                    logic signed [63:0] lo, hi;
                    on = ($signed(omem[idx]) * inv) >>> 24;         // Q16 normalized
                    // silu(z) = z * sigmoid(z), sigmoid interpolated over [-16,16]
                    zoff = $signed(zmem[idx]) + (16 <<< 12);
                    i0 = (zoff >>> 7); fr = zoff[6:0];
                    if (i0 > LUTN-1) i0 = LUTN-1;
                    lo = $signed({32'd0, slut[i0]});
                    hi = (i0 == LUTN-1) ? (64'sd1 <<< 16) : $signed({32'd0, slut[i0+1]});
                    sig  = lo + (((hi - lo) * $signed({57'd0, fr})) >>> 7);   // Q16
                    silu = ($signed(zmem[idx]) * sig) >>> 16;       // Q12
                    ow   = (on * $signed(wmem[idx])) >>> 16;        // Q16
                    yv   = (ow * silu) >>> 12;                      // Q16
                    ymem[idx] <= yv[31:0];
                    if (idx == V-1) begin idx <= 0; st <= FIN; end else idx <= idx + 1;
                end
                FIN: begin done <= 1; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
