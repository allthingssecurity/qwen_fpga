// Synthesizable int8 matvec engine with an AXI4-Lite control/data interface. This
// is the first thing to put on real silicon: the int8 GEMV + dequant datapath that
// every Qwen layer projection is built from, driven by the host over AXI-Lite. The
// host writes the activation and a weight tile into on-chip BRAM, sets the dims and
// per-row multipliers, pulses start, then reads the int16 results back. It runs the
// exact matvec_i8 that make -C rtl matvec checks bit-for-bit against the model.
//
// LANES=1: one int8 MAC per cycle. F2 fixes clk_main_a0 at 250 MHz (4 ns), so the
// per-cycle logic must be short; a single int8 multiply-accumulate closes timing
// where a 4-wide MAC carry chain did not. Address map (AXI-Lite, byte address): the
// weight BRAM is large at LANES=1 so it gets its own region at addr[20]; the small
// blocks sit in 64 KiB regions selected by addr[19:16]:
//   0x00_0000  regs   0x00 CTRL(w bit0 start) 0x04 STATUS(r done,busy) 0x08 IN 0x0C OUTROWS
//   0x01_0000  X BRAM     int8 activation, low byte of each word
//   0x02_0000  MULT BRAM  per-row int32 dequant multiplier
//   0x08_0000  Y BRAM     int16 results (read only), sign-extended
//   0x10_0000  W BRAM     int8 weights row-major, LANES int8 per word
//
// Small and BRAM-resident so the first AFI build is robust; the DDR/HBM weight
// streaming controller is the next iteration on top of this.

module qwen_matvec_engine #(
    parameter int IN_MAX  = 1024,
    parameter int OUT_MAX = 64,
    parameter int LANES   = 1
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic [31:0]  s_awaddr,
    input  logic         s_awvalid,
    output logic         s_awready,
    input  logic [31:0]  s_wdata,
    input  logic [3:0]   s_wstrb,
    input  logic         s_wvalid,
    output logic         s_wready,
    output logic [1:0]   s_bresp,
    output logic         s_bvalid,
    input  logic         s_bready,
    input  logic [31:0]  s_araddr,
    input  logic         s_arvalid,
    output logic         s_arready,
    output logic [31:0]  s_rdata,
    output logic [1:0]   s_rresp,
    output logic         s_rvalid,
    input  logic         s_rready
);
    localparam int WORDS = IN_MAX/LANES;                 // weight words per row
    localparam int WMEMN = WORDS*OUT_MAX;
    localparam int XW = $clog2(IN_MAX);
    localparam int OW = $clog2(OUT_MAX);
    localparam int WW = $clog2(WMEMN);
    localparam int CW = $clog2(WORDS);

    logic signed [7:0]      xmem [0:IN_MAX-1];
    logic signed [31:0]     mmem [0:OUT_MAX-1];
    logic [LANES*8-1:0]     wmem [0:WMEMN-1];
    logic signed [15:0]     ymem [0:OUT_MAX-1];

    logic [15:0] r_in, r_outrows, cols;
    logic        start, busy, done;

    // ---------------- AXI-Lite write ----------------
    // Robust: latch AW and W independently (a real master may present them on
    // different cycles), fire the write once both have arrived. The earlier
    // same-cycle-only handshake never started the engine on silicon.
    assign s_bresp = 2'b00;
    logic        aw_hs, w_hs;
    logic [31:0] awaddr_q, wdata_q;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_awready <= 1'b1; s_wready <= 1'b1; s_bvalid <= 1'b0;
            aw_hs <= 1'b0; w_hs <= 1'b0; start <= 1'b0; r_in <= 16'd0; r_outrows <= 16'd0;
        end else begin
            start <= 1'b0;
            if (s_awvalid && s_awready) begin awaddr_q <= s_awaddr; aw_hs <= 1'b1; s_awready <= 1'b0; end
            if (s_wvalid  && s_wready ) begin wdata_q  <= s_wdata;  w_hs  <= 1'b1; s_wready  <= 1'b0; end
            if (aw_hs && w_hs && !s_bvalid) begin
                s_bvalid <= 1'b1; aw_hs <= 1'b0; w_hs <= 1'b0;
                if (awaddr_q[20]) begin
                    wmem[awaddr_q[2+:WW]] <= wdata_q[LANES*8-1:0];   // large weight region
                end else begin
                    unique case (awaddr_q[19:16])
                        4'h0: unique case (awaddr_q[7:2])
                                  6'd0: if (wdata_q[0]) start <= 1'b1;
                                  6'd2: r_in      <= wdata_q[15:0];
                                  6'd3: r_outrows <= wdata_q[15:0];
                                  default: ;
                              endcase
                        4'h1: xmem[awaddr_q[2+:XW]] <= wdata_q[7:0];
                        4'h2: mmem[awaddr_q[2+:OW]] <= wdata_q;
                        default: ;
                    endcase
                end
            end
            if (s_bvalid && s_bready) begin s_bvalid <= 1'b0; s_awready <= 1'b1; s_wready <= 1'b1; end
        end
    end

    // ---------------- AXI-Lite read ----------------
    // 0x04 status {state,busy,done}, 0x08 readback r_in, 0x0C readback r_outrows
    // (readbacks confirm writes land on hardware), 0x8_xxxx results.
    assign s_arready = !s_rvalid;
    assign s_rresp   = 2'b00;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_rvalid <= 1'b0; s_rdata <= 32'd0;
        end else begin
            if (s_arvalid && s_arready) begin
                s_rvalid <= 1'b1;
                unique case (s_araddr[19:16])
                    4'h0: unique case (s_araddr[7:2])
                              6'd1: s_rdata <= {30'd0, busy, done};
                              6'd2: s_rdata <= {16'd0, r_in};        // readback confirms writes land
                              6'd3: s_rdata <= {16'd0, r_outrows};
                              default: s_rdata <= 32'hCAFE0000;      // sentinel: reads reach the engine
                          endcase
                    4'h8: s_rdata <= {{16{ymem[s_araddr[2+:OW]][15]}}, ymem[s_araddr[2+:OW]]};
                    default: s_rdata <= 32'd0;
                endcase
            end else if (s_rvalid && s_rready) begin
                s_rvalid <= 1'b0;
            end
        end
    end

    // ---------------- matvec sequencer ----------------
    // The whole IN_MAX activation is loaded into the GEMV (it reads xmem by column
    // during streaming), then each row streams WORDS weight words with its dequant
    // multiplier held, and WAITY catches the acc_vld the GEMV raises one cycle after
    // the row's last word. w_en is low in WAITY so the GEMV's column counter, which
    // auto-segments rows, stays aligned.
    typedef enum logic [2:0] {IDLE, LOADX, STREAM, WAITY, FINISH} st_t;
    st_t state;
    logic [OW-1:0] row;
    logic [XW-1:0] xi;
    logic [CW-1:0] col;
    logic          mv_load, mv_wen, mv_rst;
    logic signed [7:0]  mv_xbyte;
    logic [LANES*8-1:0] mv_wword;
    logic signed [31:0] mv_mult;
    logic signed [15:0] mv_y;
    logic               mv_yvld;

    matvec_i8 #(.IN(IN_MAX), .OUT(OUT_MAX), .LANES(LANES)) u_mv (
        .clk(clk), .rst(mv_rst),
        .load_en(mv_load), .x_byte(mv_xbyte),
        .w_en(mv_wen), .w_word(mv_wword),
        .mult(mv_mult), .y_q(mv_y), .y_vld(mv_yvld)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; busy <= 1'b0; done <= 1'b0;
            mv_load <= 1'b0; mv_wen <= 1'b0; mv_rst <= 1'b1;
            row <= '0; xi <= '0; col <= '0;
        end else begin
            mv_load <= 1'b0; mv_wen <= 1'b0; mv_rst <= 1'b0;
            unique case (state)
                IDLE: if (start) begin
                    busy <= 1'b1; done <= 1'b0; mv_rst <= 1'b1; xi <= '0; state <= LOADX;
                end
                LOADX: begin                              // load all IN_MAX activations
                    mv_load <= 1'b1; mv_xbyte <= xmem[xi];
                    if (xi == XW'(IN_MAX-1)) begin xi <= '0; row <= '0; col <= '0; state <= STREAM; end
                    else xi <= xi + 1'b1;
                end
                STREAM: begin                             // one row of WORDS weight words
                    mv_wen <= 1'b1; mv_wword <= wmem[{row, col}]; mv_mult <= mmem[row];
                    if (col == CW'(WORDS-1)) begin col <= '0; state <= WAITY; end
                    else col <= col + 1'b1;
                end
                WAITY: begin                              // catch acc_vld, multiplier held
                    if (mv_yvld) begin
                        ymem[row] <= mv_y;
                        if (row == OW'(r_outrows[OW-1:0]-1)) begin busy <= 1'b0; done <= 1'b1; state <= IDLE; end
                        else begin row <= row + 1'b1; state <= STREAM; end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
