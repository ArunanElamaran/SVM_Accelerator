// non-ANSI port definitions to use params in definitions
module svm (clk, rst, x0, x1, label);

    // size parameters
    localparam INT_BITS = 5;
    localparam FRAC_BITS = 2;
    localparam SIZE = INT_BITS + FRAC_BITS;
	 
    // port definitions (2s complement)
    input logic clk;
    input logic rst;
    input logic signed [SIZE-1:0] x0;
    input logic signed [SIZE-1:0] x1;
    output logic label;

    // intermediate values (2s complement)
    logic signed [SIZE*2-1:0] val1;
    logic signed [SIZE*2-1:0] val2;
    logic signed [SIZE*2-1:0] val3;
    // logic signed [SIZE*2+1:0] val4;
    logic signed [SIZE*2:0] val5;
    logic signed [SIZE*2+1:0] val6;

    // combinational logic
    always_comb begin
        // val1 = 18'sb000001110010111110 * x0; //w0
        // val2 = 18'sb111101111100100111 * x1; //w1
        // val3 = {{INT_BITS{1'b1}}, 18'sb111100100011111110, {FRAC_BITS{1'b0}}}; //b
        // val3 = {{INT_BITS{1'b1}}, 7'sb1110010, {FRAC_BITS{1'b0}}}; //b
        // val4 = val1 + val2 + val3;
        val6 = val5 + val3;
    end

    // sequential logic
    always_ff @(posedge clk) begin
        if (rst) begin
            label <= 1'b0;
        end
        else begin
            // cycle 1 multiplications
            val1 <= 7'sb0000111 * x0; //w0
            val2 <= 7'sb1111000 * x1; //w1

            // cycle 2 addition and bit extension
            val3 <= {{INT_BITS{1'b1}}, 7'sb1110010, {FRAC_BITS{1'b0}}}; //b
            val5 <= val1 + val2;

            // cycle 3 addition and bit select MSB for pos/neg detection
            label <= val6[SIZE*2] ? 1'b0 : 1'b1;
        end
    end

endmodule