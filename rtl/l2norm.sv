// L2 normalize over K elements, then a post-scale: y = x*rsqrt(sum(x^2)+eps)*scale.
// Used on q (scale = 1/sqrt(128)) and k (scale = 1) in DeltaNet. Same integer
// reciprocal square root as rmsnorm (isqrt + divide, no floating-point unit).
// Q16 activations; scale is Q16. Checked in Verilator against golden l2norm.

module l2norm #(
    parameter int N = 128
) (
    input  logic               clk,
    input  logic               rst,
    input  logic               we_x,
    input  logic [7:0]         addr_x,
    input  logic signed [31:0] din_x,
    input  logic signed [31:0] scale,   // Q16 post-scale
    input  logic               start,
    output logic               done,
    input  logic [7:0]         raddr_y,
    output logic signed [31:0] rdata_y
);
    logic signed [31:0] xmem [0:N-1];
    logic signed [31:0] ymem [0:N-1];
    assign rdata_y = ymem[raddr_y];
    localparam logic [63:0] EPS_Q32 = 64'd4295;

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
    logic signed [63:0] ss, inv;

    always_ff @(posedge clk) begin
        if (rst) begin st <= IDLE; done <= 0; idx <= 0; ss <= 0; end
        else begin
            done <= 0;
            if (st == IDLE && we_x) xmem[addr_x] <= din_x;
            case (st)
                IDLE: if (start) begin ss <= 0; idx <= 0; st <= ACC; end
                ACC: begin
                    logic signed [63:0] xv; xv = $signed(xmem[idx]);
                    ss <= ss + (xv * xv);                          // Q32 (sum, no /N)
                    if (idx == N-1) begin idx <= 0; st <= RSQ; end else idx <= idx + 1;
                end
                RSQ: begin
                    logic [63:0] r; logic signed [63:0] a;
                    a = ss + EPS_Q32;
                    r = isqrt(a << 16);
                    inv <= (64'sd1 <<< 40) / $signed(r);           // Q16
                    st <= OUT;
                end
                OUT: begin
                    logic signed [63:0] xn, yv;
                    xn = ($signed(xmem[idx]) * inv) >>> 16;        // normalized, Q16
                    yv = (xn * $signed(scale)) >>> 16;             // Q16
                    ymem[idx] <= yv[31:0];
                    if (idx == N-1) begin idx <= 0; st <= FIN; end else idx <= idx + 1;
                end
                FIN: begin done <= 1; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
