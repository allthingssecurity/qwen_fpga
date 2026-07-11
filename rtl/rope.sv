// Partial rotary position embedding. Only the first RD (64) of HD (256) dims
// rotate; the rest pass through. For i < RD:
//   out[i] = q[i]*cos[i] + rotate_half(q)[i]*sin[i]
// where rotate_half is [-q[RD/2 : RD], q[0 : RD/2]]. cos/sin are Q14 (in [-1,1]),
// q and out are Q12. Checked in Verilator against golden apply_rope.

module rope #(
    parameter int HD = 256,
    parameter int RD = 64,
    parameter int Q  = 12,   // q/out fractional bits
    parameter int QC = 14    // cos/sin fractional bits
) (
    input  logic               clk,
    input  logic               rst,
    input  logic               we_q,
    input  logic [8:0]         addr_q,
    input  logic signed [31:0] din_q,
    input  logic               we_cs,      // load cos then sin, addr in [0,RD) cos, [RD,2RD) sin
    input  logic [8:0]         addr_cs,
    input  logic signed [31:0] din_cs,
    input  logic               start,
    output logic               done,
    input  logic [8:0]         raddr_o,
    output logic signed [31:0] rdata_o
) ;
    logic signed [31:0] qmem [0:HD-1];
    logic signed [31:0] cmem [0:RD-1];
    logic signed [31:0] smem [0:RD-1];
    logic signed [31:0] omem [0:HD-1];
    assign rdata_o = omem[raddr_o];

    typedef enum logic [1:0] {IDLE, OUT, FIN} state_t;
    state_t st;
    logic [8:0] idx;

    always_ff @(posedge clk) begin
        if (rst) begin st <= IDLE; done <= 0; idx <= 0; end
        else begin
            done <= 0;
            if (st == IDLE) begin
                if (we_q) qmem[addr_q] <= din_q;
                if (we_cs) begin
                    if (addr_cs < RD) cmem[addr_cs]      <= din_cs;
                    else              smem[addr_cs - RD] <= din_cs;
                end
            end
            case (st)
                IDLE: if (start) begin idx <= 0; st <= OUT; end

                OUT: begin
                    if (idx >= RD) begin
                        omem[idx] <= qmem[idx];              // pass-through
                    end else begin
                        logic signed [31:0] h;
                        logic signed [63:0] acc;
                        if (idx < RD/2) h = -$signed(qmem[idx + RD/2]);
                        else            h =  $signed(qmem[idx - RD/2]);
                        acc = $signed(qmem[idx]) * $signed(cmem[idx])
                            + h * $signed(smem[idx]);         // Q(Q+QC)
                        omem[idx] <= acc >>> QC;              // back to Q12
                    end
                    if (idx == HD-1) begin idx <= 0; st <= FIN; end
                    else idx <= idx + 1;
                end

                FIN: begin done <= 1; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
