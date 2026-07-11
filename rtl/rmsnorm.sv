// RMSNorm in fixed point, no floating-point unit. Computes
//   y[i] = x[i] * rsqrt(mean(x^2) + eps) * (1 + w[i])
// which is the Qwen3.5 convention (gamma stored centered at zero, so the gain is
// 1 + w). Activations and weights are Q16 int32.
//
// The reciprocal square root is done with an integer square root plus a divide.
// With mean represented as a Q32 value a, sqrt(a_real) = isqrt(a<<16) / 2^24, so
//   inv (Q16) = 2^40 / isqrt(a << 16).
// The isqrt is the classic bit-by-bit method. The divide and isqrt are shown
// combinational here to prove the arithmetic; both become iterative/pipelined for
// timing, which does not change the result.
//
// Checked in Verilator against the golden rmsnorm (rtl/tb/sim_rmsnorm.cpp).

module rmsnorm #(
    parameter int N   = 1024,
    parameter int Q   = 16,
    parameter int LGN = 10    // log2(N)
) (
    input  logic               clk,
    input  logic               rst,
    input  logic               we_x,
    input  logic [11:0]        addr_x,
    input  logic signed [31:0] din_x,
    input  logic               we_w,
    input  logic [11:0]        addr_w,
    input  logic signed [31:0] din_w,
    input  logic               start,
    output logic               done,
    input  logic [11:0]        raddr_y,
    output logic signed [31:0] rdata_y
);
    logic signed [31:0] xmem [0:N-1];
    logic signed [31:0] wmem [0:N-1];
    logic signed [31:0] ymem [0:N-1];
    assign rdata_y = ymem[raddr_y];

    localparam logic [63:0] EPS_Q32 = 64'd4295;     // round(1e-6 * 2^32)

    // combinational integer sqrt of a 64-bit value (floor)
    function automatic logic [63:0] isqrt (input logic [63:0] a);
        logic [63:0] rem, root, bitv;
        rem = a; root = 0; bitv = 64'h4000_0000_0000_0000;
        for (int s = 0; s < 32; s++) begin
            if (bitv > rem) bitv = bitv >> 2;
        end
        for (int s = 0; s < 32; s++) begin
            if (bitv != 0) begin
                if (rem >= (root + bitv)) begin
                    rem  = rem - (root + bitv);
                    root = (root >> 1) + bitv;
                end else begin
                    root = root >> 1;
                end
                bitv = bitv >> 2;
            end
        end
        return root;
    endfunction

    typedef enum logic [2:0] {IDLE, ACC, RSQ, OUT, FIN} state_t;
    state_t st;
    logic [11:0] idx;
    logic signed [63:0] ss;
    logic signed [63:0] inv;   // Q16

    always_ff @(posedge clk) begin
        if (rst) begin
            st <= IDLE; done <= 0; idx <= 0; ss <= 0; inv <= 0;
        end else begin
            done <= 0;
            if (st == IDLE) begin
                if (we_x) xmem[addr_x] <= din_x;
                if (we_w) wmem[addr_w] <= din_w;
            end
            case (st)
                IDLE: if (start) begin ss <= 0; idx <= 0; st <= ACC; end

                ACC: begin  // ss += x[idx]^2  (Q16^2 = Q32)
                    logic signed [63:0] xv;
                    xv = $signed(xmem[idx]);
                    ss <= ss + (xv * xv);
                    if (idx == N-1) begin idx <= 0; st <= RSQ; end
                    else idx <= idx + 1;
                end

                RSQ: begin  // a = mean + eps ; inv = 2^40 / isqrt(a<<16)
                    logic signed [63:0] a;
                    logic [63:0] r;
                    a = (ss >>> LGN) + EPS_Q32;         // Q32
                    r = isqrt(a << 16);                 // sqrt(a_real) * 2^24
                    inv <= (64'sd1 <<< 40) / $signed(r);  // Q16
                    st <= OUT;
                end

                OUT: begin  // y = (x*inv >> Q) * (1+w) >> Q
                    logic signed [63:0] xn, onepw, yv;
                    xn    = ($signed(xmem[idx]) * inv) >>> Q;        // Q16 normalized x
                    onepw = (64'sd1 <<< Q) + $signed(wmem[idx]);     // (1+w) in Q16
                    yv    = (xn * onepw) >>> Q;                      // Q16
                    ymem[idx] <= yv[31:0];
                    if (idx == N-1) begin idx <= 0; st <= FIN; end
                    else idx <= idx + 1;
                end

                FIN: begin done <= 1; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
