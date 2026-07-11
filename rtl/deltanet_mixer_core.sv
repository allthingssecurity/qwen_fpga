// DeltaNet core for one head, composed in RTL: gate_math -> deltanet_head ->
// gated_norm, wired by a controller. This is a real hardware composition; the
// fixed-point formats line up by design (gate emits gexp Q30 / beta Q24, which
// the recurrence consumes; the recurrence emits o Q24, which the gated norm
// consumes). The controller sequences each block's start/done and copies the
// recurrence output into the norm.
//
// Verified end to end against golden for head 0 (rtl/tb/sim_mixer.cpp): given the
// gate inputs, the loaded q,k,v,state, and w,z, the final y matches the model.

module deltanet_mixer_core #(
    parameter int K = 128,
    parameter int V = 128
) (
    input  logic               clk,
    input  logic               rst,
    // gate lookup tables + head inputs
    input  logic               we_gsp, we_gex, we_gsg,
    input  logic [9:0]         g_addr,
    input  logic [31:0]        g_din,
    input  logic signed [31:0] gm_a, gm_dt, gm_A, gm_b,
    // recurrence loads
    input  logic               we_s, we_q, we_k, we_v,
    input  logic [13:0]        addr_s,
    input  logic [6:0]         addr_qkv,
    input  logic signed [31:0] din_s, din_q, din_k, din_v,
    // gated-norm loads (sigmoid lut, w, z)
    input  logic               we_nlut, we_w, we_z,
    input  logic [9:0]         n_addr,
    input  logic [31:0]        n_din,
    input  logic [7:0]         addr_wz,
    input  logic signed [31:0] din_w, din_z,
    // control + result
    input  logic               start,
    output logic               done,
    input  logic [7:0]         raddr_y,
    output logic signed [31:0] rdata_y
);
    // ---- sub-blocks
    logic        gm_ivld, gm_ovld;
    logic [31:0] gm_gexp, gm_beta;
    gate_math u_gate (
        .clk(clk), .rst(rst),
        .we_sp(we_gsp), .we_ex(we_gex), .we_sg(we_gsg), .addr_lut(g_addr), .din_lut(g_din),
        .in_vld(gm_ivld), .a(gm_a), .dt(gm_dt), .A(gm_A), .b(gm_b),
        .gexp(gm_gexp), .beta(gm_beta), .out_vld(gm_ovld)
    );

    logic [31:0] r_gexp, r_beta;   // latched gate outputs
    logic        d_start, d_busy, d_done;
    logic [13:0] d_raddr_s; logic signed [31:0] d_rdata_s;
    logic [6:0]  d_raddr_o; logic signed [31:0] d_rdata_o;
    deltanet_head u_delta (
        .clk(clk), .rst(rst),
        .we_s(we_s), .addr_s(addr_s), .din_s(din_s),
        .we_q(we_q), .addr_q(addr_qkv), .din_q(din_q),
        .we_k(we_k), .addr_k(addr_qkv), .din_k(din_k),
        .we_v(we_v), .addr_v(addr_qkv), .din_v(din_v),
        .gexp(r_gexp), .beta(r_beta),
        .start(d_start), .busy(d_busy), .done(d_done),
        .raddr_s(d_raddr_s), .rdata_s(d_rdata_s),
        .raddr_o(d_raddr_o), .rdata_o(d_rdata_o)
    );

    logic        n_we_o; logic [7:0] n_addr_o; logic signed [31:0] n_din_o;
    logic        n_start, n_done;
    gated_norm u_gnorm (
        .clk(clk), .rst(rst),
        .we_lut(we_nlut), .addr_lut(n_addr), .din_lut(n_din),
        .we_o(n_we_o), .addr_o(n_addr_o), .din_o(n_din_o),
        .we_w(we_w), .addr_w(addr_wz), .din_w(din_w),
        .we_z(we_z), .addr_z(addr_wz), .din_z(din_z),
        .start(n_start), .done(n_done),
        .raddr_y(raddr_y), .rdata_y(rdata_y)
    );

    // ---- controller
    typedef enum logic [3:0] {IDLE, GATE, GWAIT, DSTART, DWAIT, COPY, NSTART, NWAIT, DONE_S} st_t;
    st_t st;
    logic [7:0] ci;

    // copy datapath is combinational so o[ci] is read and written the same cycle
    assign d_raddr_o = ci[6:0];
    assign n_we_o    = (st == COPY);
    assign n_addr_o  = ci;
    assign n_din_o   = d_rdata_o;

    always_ff @(posedge clk) begin
        if (rst) begin
            st <= IDLE; done <= 0; gm_ivld <= 0; d_start <= 0; n_start <= 0; ci <= 0;
        end else begin
            done <= 0; gm_ivld <= 0; d_start <= 0; n_start <= 0;
            case (st)
                IDLE:  if (start) begin gm_ivld <= 1; st <= GATE; end
                GATE:  st <= GWAIT;                               // gate latency
                GWAIT: begin r_gexp <= gm_gexp; r_beta <= gm_beta; st <= DSTART; end
                DSTART: begin d_start <= 1; st <= DWAIT; end
                DWAIT: if (d_done) begin ci <= 0; st <= COPY; end
                COPY:  if (ci == V-1) st <= NSTART; else ci <= ci + 1;
                NSTART: begin n_start <= 1; st <= NWAIT; end
                NWAIT: if (n_done) st <= DONE_S;
                DONE_S: begin done <= 1; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
