// Gated DeltaNet recurrence, one head, in fixed point. This is the distinctive
// part of Qwen3.5, a running state S[K][V] updated per token. The golden model
// keeps S in fp32; here it is Q(QS) fixed point so it is real integer hardware,
// no floating-point unit.
//
// Per decode step, given q,k,v (Q24), a scalar decay gexp = exp(g) (Q30), and a
// scalar beta (Q24):
//   S   = S * gexp                 decay the state
//   kv  = k^T S                    contract over K
//   d   = (v - kv) * beta          the delta
//   S   = S + k (x) d              rank-1 update
//   o   = q^T S                    read out
//
// Done as a simple FSM, one multiply-accumulate per cycle, so it is synthesizable
// in structure. Parallelism and BRAM pipelining are later refinements; this
// proves the fixed-point datapath matches the model. Checked in Verilator against
// the warm-state golden vectors (rtl/tb/sim_deltanet.cpp).

module deltanet_head #(
    parameter int K  = 128,
    parameter int V  = 128,
    parameter int QS = 24,   // fractional bits for q,k,v,S,delta,o,beta,kv
    parameter int QG = 30    // fractional bits for gexp (in (0,1])
) (
    input  logic                clk,
    input  logic                rst,
    // load ports (Q24 fixed point), used while idle
    input  logic                we_s,
    input  logic [13:0]         addr_s,
    input  logic signed [31:0]  din_s,
    input  logic                we_q,
    input  logic [6:0]          addr_q,
    input  logic signed [31:0]  din_q,
    input  logic                we_k,
    input  logic [6:0]          addr_k,
    input  logic signed [31:0]  din_k,
    input  logic                we_v,
    input  logic [6:0]          addr_v,
    input  logic signed [31:0]  din_v,
    input  logic signed [31:0]  gexp,    // Q30
    input  logic signed [31:0]  beta,    // Q24
    // control
    input  logic                start,
    output logic                busy,
    output logic                done,
    // read back (Q24)
    input  logic [13:0]         raddr_s,
    output logic signed [31:0]  rdata_s,
    input  logic [6:0]          raddr_o,
    output logic signed [31:0]  rdata_o
);
    logic signed [31:0] Smem [0:K*V-1];
    logic signed [31:0] qmem [0:K-1];
    logic signed [31:0] kmem [0:K-1];
    logic signed [31:0] vmem [0:V-1];
    logic signed [31:0] omem [0:V-1];
    logic signed [31:0] dmem [0:V-1];   // delta
    logic signed [63:0] kv  [0:V-1];    // k^T S accumulator (Q40 before shift)
    logic signed [63:0] oacc[0:V-1];    // q^T S accumulator (Q40 before shift)

    assign rdata_s = Smem[raddr_s];
    assign rdata_o = omem[raddr_o];

    typedef enum logic [2:0] {IDLE, P1, P2, P3, P4, FIN} state_t;
    state_t st;
    logic [6:0] kk, vv;

    always_ff @(posedge clk) begin
        if (rst) begin
            st <= IDLE; busy <= 0; done <= 0; kk <= 0; vv <= 0;
        end else begin
            done <= 0;
            // loads only while idle
            if (st == IDLE) begin
                if (we_s) Smem[addr_s] <= din_s;
                if (we_q) qmem[addr_q] <= din_q;
                if (we_k) kmem[addr_k] <= din_k;
                if (we_v) vmem[addr_v] <= din_v;
            end

            case (st)
                IDLE: if (start) begin
                    busy <= 1; kk <= 0; vv <= 0;
                    for (int i = 0; i < V; i++) kv[i] <= 64'sd0;
                    st <= P1;
                end

                // pass 1: S <- S*gexp, and kv[vv] += (S*gexp) * k[kk]
                // all products forced to 64-bit by sign-extending operands first
                P1: begin
                    logic signed [63:0] sm, ge, prod, s_dec, kf, kprod;
                    sm     = $signed(Smem[{kk,vv}]);   // Q24
                    ge     = $signed(gexp);            // Q30
                    prod   = sm * ge;                  // Q50
                    s_dec  = prod >>> QG;              // Q24
                    kf     = $signed(kmem[kk]);        // Q24
                    kprod  = s_dec * kf;               // Q40
                    Smem[{kk,vv}] <= s_dec[31:0];
                    kv[vv] <= kv[vv] + kprod;
                    if (vv == V-1) begin
                        vv <= 0;
                        if (kk == K-1) begin kk <= 0; st <= P2; end
                        else kk <= kk + 1;
                    end else vv <= vv + 1;
                end

                // pass 2: delta[vv] = (v[vv] - kv[vv]) * beta ; clear o accumulator
                P2: begin
                    logic signed [63:0] kvq, vf, bf, diff, dprod;
                    kvq    = kv[vv] >>> QS;             // Q24
                    vf     = $signed(vmem[vv]);        // Q24
                    bf     = $signed(beta);            // Q24
                    diff   = vf - kvq;                 // Q24
                    dprod  = diff * bf;                // Q40
                    dmem[vv] <= (dprod >>> QS);         // Q24
                    oacc[vv] <= 64'sd0;
                    if (vv == V-1) begin vv <= 0; st <= P3; end
                    else vv <= vv + 1;
                end

                // pass 3: S <- S + k[kk]*delta[vv] ; oacc[vv] += S_new * q[kk]
                P3: begin
                    logic signed [63:0] kf, df, upd, qf, oprod;
                    logic signed [31:0] s_new;
                    kf     = $signed(kmem[kk]);        // Q24
                    df     = $signed(dmem[vv]);        // Q24
                    upd    = (kf * df) >>> QS;          // Q24
                    s_new  = $signed(Smem[{kk,vv}]) + upd[31:0];
                    qf     = $signed(qmem[kk]);        // Q24
                    oprod  = $signed(s_new) * qf;      // Q40
                    Smem[{kk,vv}] <= s_new;
                    oacc[vv] <= oacc[vv] + oprod;
                    if (vv == V-1) begin
                        vv <= 0;
                        if (kk == K-1) begin kk <= 0; st <= P4; end
                        else kk <= kk + 1;
                    end else vv <= vv + 1;
                end

                // pass 4: o[vv] = oacc[vv] >> QS
                P4: begin
                    omem[vv] <= oacc[vv] >>> QS;
                    if (vv == V-1) begin vv <= 0; st <= FIN; end
                    else vv <= vv + 1;
                end

                FIN: begin busy <= 0; done <= 1; st <= IDLE; end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
