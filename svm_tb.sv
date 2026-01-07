//`include "svm_top.sv"

module svm_tb(
    output logic done
);

    logic clk = 0;
    logic rst = 0;
    logic start = 0;

    svm_top dut(.clk(clk), .rst(rst), .start(start), .done(done));

    always #1 clk = ~clk;

    initial begin
        // $dumpfile("dump.vcd");
        // $dumpvars(0, svm_tb);

        #1 
        rst = 0; // ACTIVE LOW

        #2 
        rst = 1;

        // get data from rom
        #2;
        start = 0; // ACTIVE LOW

        // process
        #2;
        start = 1;

        #10 $stop;


    end

endmodule