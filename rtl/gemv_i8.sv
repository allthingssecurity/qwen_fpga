// int8 GEMV, the workhorse of Qwen decode. Computes acc[o] = sum_i w[o][i]*x[i]
// with int8 weights and int8 activations, accumulating in int32. This is the
// exact integer core; the fp32 dequant (acc * sx * sw[o]) is a later stage.
//
// Weight-stationary streaming: the activation vector is loaded on chip once,
// then the weights stream in LANES int8 per cycle in row-major order. Every
// IN/LANES cycles one output row finishes and acc_o is emitted. This is the
// shape that maps onto HBM: weights flow past a small resident activation.
//
// Verified in simulation against the golden int8 model with rtl/tb/sim_gemv.cpp,
// exact match required since the math is pure integer.

module gemv_i8 #(
    parameter int IN    = 1024,   // input dim
    parameter int OUT   = 512,    // output dim (rows)
    parameter int LANES = 8       // int8 weights consumed per cycle (64-bit word)
) (
    input  logic                     clk,
    input  logic                     rst,
    // load the activation vector, one int8 per cycle
    input  logic                     load_en,
    input  logic signed [7:0]        x_byte,
    // stream weights, LANES int8 per cycle, row-major
    input  logic                     w_en,
    input  logic [LANES*8-1:0]       w_word,
    // one int32 accumulator per completed output row
    output logic signed [31:0]       acc_o,
    output logic                     acc_vld
);
    localparam int WPR = IN / LANES;   // weight words per output row

    logic signed [7:0] xmem [0:IN-1];  // resident activation (int8)
    int unsigned load_ptr;             // sim-sized counters; sized down for synth
    int unsigned col;                  // word index within the current row
    logic signed [31:0] acc;

    always_ff @(posedge clk) begin
        if (rst) begin
            load_ptr <= 0;
            col      <= 0;
            acc      <= 0;
            acc_o    <= 0;
            acc_vld  <= 1'b0;
        end else begin
            acc_vld <= 1'b0;

            if (load_en) begin
                xmem[load_ptr] <= x_byte;
                load_ptr <= load_ptr + 1;
            end

            if (w_en) begin
                // LANES int8 x int8 products, summed this cycle
                logic signed [31:0] partial;
                partial = '0;
                for (int j = 0; j < LANES; j++) begin
                    logic signed [7:0] wj;
                    logic signed [7:0] xj;
                    wj = w_word[j*8 +: 8];
                    xj = xmem[col*LANES + j];
                    partial = partial + ($signed(wj) * $signed(xj));
                end

                if (col == WPR-1) begin
                    acc_o   <= acc + partial;   // last word of the row
                    acc_vld <= 1'b1;
                    acc     <= 0;
                    col     <= 0;
                end else begin
                    acc <= acc + partial;
                    col <= col + 1;
                end
            end
        end
    end
endmodule
