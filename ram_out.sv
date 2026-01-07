module ram_out(
    input logic clk,
    input logic wr_en,
    input logic [4:0] ram_addr,
    input logic [29:0] data_in,
    output logic [29:0] data_out
);

    reg [29:0] mem [31:0] /* synthesis ramstyle = M9K */;

    always @(posedge clk) begin
        if (wr_en == 1'b1) begin
            mem[ram_addr] <= data_in; // write
        end
        
        data_out <= mem[ram_addr]; // read
    end

endmodule