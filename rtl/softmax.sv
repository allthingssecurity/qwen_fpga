// Softmax over T scores, fixed point, no floating-point unit. Three passes:
//   1. find the max score
//   2. e_t = exp(s_t - max) via an exp lookup table (covers [-16, 0], Q16)
//      and accumulate the sum
//   3. w_t = e_t / sum
// The exp table and a divide are the only non-trivial ops, both already used in
// other modules. Scores are Q10; weights come out in Q16 (a probability in [0,1]).
//
// Checked in Verilator against numpy softmax.

module softmax #(
    parameter int TMAX = 4096,
    parameter int QS   = 10,     // score fractional bits
    parameter int LUTN = 1024,   // exp table entries over [-16, 0]
    parameter int SH   = 4       // QS - log2(LUTN/16): maps (s-max) Q10 -> index
) (
    input  logic               clk,
    input  logic               rst,
    input  logic               we_lut,
    input  logic [9:0]         addr_lut,
    input  logic [31:0]        din_lut,     // exp value, Q16
    input  logic               we_s,
    input  logic [11:0]        addr_s,
    input  logic signed [31:0] din_s,       // score, Q10
    input  logic [11:0]        n_scores,
    input  logic               start,
    output logic               done,
    input  logic [11:0]        raddr_w,
    output logic [31:0]        rdata_w       // weight, Q16
);
    logic [31:0]        elut [0:LUTN-1];
    logic signed [31:0] smem [0:TMAX-1];
    logic [31:0]        emem [0:TMAX-1];   // e_t, Q16
    logic [31:0]        wmem [0:TMAX-1];   // w_t, Q16
    assign rdata_w = wmem[raddr_w];

    typedef enum logic [2:0] {IDLE, MAX, EXP, NORM, FIN} state_t;
    state_t st;
    logic [11:0] idx;
    logic signed [31:0] smax;
    logic [63:0] esum;

    always_ff @(posedge clk) begin
        if (rst) begin st <= IDLE; done <= 0; idx <= 0; esum <= 0; end
        else begin
            done <= 0;
            if (st == IDLE) begin
                if (we_lut) elut[addr_lut] <= din_lut;
                if (we_s)   smem[addr_s]   <= din_s;
            end
            case (st)
                IDLE: if (start) begin idx <= 0; smax <= -32'sd2147483647; st <= MAX; end

                MAX: begin
                    if ($signed(smem[idx]) > smax) smax <= $signed(smem[idx]);
                    if (idx == n_scores-1) begin idx <= 0; esum <= 0; st <= EXP; end
                    else idx <= idx + 1;
                end

                EXP: begin  // e = exp(s - max) via LUT ; esum += e
                    logic signed [31:0] d, ix;
                    logic [9:0] li;
                    logic [31:0] ev;
                    d  = $signed(smem[idx]) - smax;   // <= 0, Q10
                    ix = (d >>> SH) + LUTN;           // (d + 16*2^QS) -> [0,LUTN)
                    if (ix < 0)            li = 0;
                    else if (ix > LUTN-1)  li = LUTN-1;
                    else                   li = ix[9:0];
                    ev = elut[li];
                    emem[idx] <= ev;
                    esum <= esum + ev;
                    if (idx == n_scores-1) begin idx <= 0; st <= NORM; end
                    else idx <= idx + 1;
                end

                NORM: begin  // w = e / sum  (Q16)
                    logic [63:0] num;
                    num = {emem[idx], 16'd0};          // e << 16
                    wmem[idx] <= num / esum;
                    if (idx == n_scores-1) begin idx <= 0; st <= FIN; end
                    else idx <= idx + 1;
                end

                FIN: begin done <= 1; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
