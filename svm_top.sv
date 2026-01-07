module svm_top (
    input logic clk,
    input logic rst, 
    input logic start,
    output logic done
);
	
    // state machine names
    // typedef enum {IDLE, STARTING, PROCESSING} state_e;
    typedef enum {IDLE, STARTING, PROCESSING1, PROCESSING2, PROCESSING3} state_e;
    state_e state;

    // control signals + data signals
    logic ready;
    logic wr_en;
    logic [4:0] addr, ram_addr;
    logic signed [6:0] x0_array [31:0];
    logic signed [6:0] x1_array [31:0];
    logic label_array [29:0];
    logic [29:0] data_in, label_out;
	logic clk_250M;
	 
	pll pll0(.inclk0(clk), .c0(clk_250M)); //10MHz to 250MHz PLL
    //assign clk_250M = clk;

    // ROM and RAM instantiations
    rom_x0_0 rom_x0_0_inst (.clk(clk_250M), .addr(addr), .q(x0_array[0]));
    rom_x1_0 rom_x1_0_inst (.clk(clk_250M), .addr(addr), .q(x1_array[0]));
    rom_x0_1 rom_x0_1_inst (.clk(clk_250M), .addr(addr), .q(x0_array[1]));
    rom_x1_1 rom_x1_1_inst (.clk(clk_250M), .addr(addr), .q(x1_array[1]));
    rom_x0_2 rom_x0_2_inst (.clk(clk_250M), .addr(addr), .q(x0_array[2]));
    rom_x1_2 rom_x1_2_inst (.clk(clk_250M), .addr(addr), .q(x1_array[2]));
    rom_x0_3 rom_x0_3_inst (.clk(clk_250M), .addr(addr), .q(x0_array[3]));
    rom_x1_3 rom_x1_3_inst (.clk(clk_250M), .addr(addr), .q(x1_array[3]));
    rom_x0_4 rom_x0_4_inst (.clk(clk_250M), .addr(addr), .q(x0_array[4]));
    rom_x1_4 rom_x1_4_inst (.clk(clk_250M), .addr(addr), .q(x1_array[4]));
    rom_x0_5 rom_x0_5_inst (.clk(clk_250M), .addr(addr), .q(x0_array[5]));
    rom_x1_5 rom_x1_5_inst (.clk(clk_250M), .addr(addr), .q(x1_array[5]));
    rom_x0_6 rom_x0_6_inst (.clk(clk_250M), .addr(addr), .q(x0_array[6]));
    rom_x1_6 rom_x1_6_inst (.clk(clk_250M), .addr(addr), .q(x1_array[6]));
    rom_x0_7 rom_x0_7_inst (.clk(clk_250M), .addr(addr), .q(x0_array[7]));
    rom_x1_7 rom_x1_7_inst (.clk(clk_250M), .addr(addr), .q(x1_array[7]));
    rom_x0_8 rom_x0_8_inst (.clk(clk_250M), .addr(addr), .q(x0_array[8]));
    rom_x1_8 rom_x1_8_inst (.clk(clk_250M), .addr(addr), .q(x1_array[8]));
    rom_x0_9 rom_x0_9_inst (.clk(clk_250M), .addr(addr), .q(x0_array[9]));
    rom_x1_9 rom_x1_9_inst (.clk(clk_250M), .addr(addr), .q(x1_array[9]));
    rom_x0_10 rom_x0_10_inst (.clk(clk_250M), .addr(addr), .q(x0_array[10]));
    rom_x1_10 rom_x1_10_inst (.clk(clk_250M), .addr(addr), .q(x1_array[10]));
    rom_x0_11 rom_x0_11_inst (.clk(clk_250M), .addr(addr), .q(x0_array[11]));
    rom_x1_11 rom_x1_11_inst (.clk(clk_250M), .addr(addr), .q(x1_array[11]));
    rom_x0_12 rom_x0_12_inst (.clk(clk_250M), .addr(addr), .q(x0_array[12]));
    rom_x1_12 rom_x1_12_inst (.clk(clk_250M), .addr(addr), .q(x1_array[12]));
    rom_x0_13 rom_x0_13_inst (.clk(clk_250M), .addr(addr), .q(x0_array[13]));
    rom_x1_13 rom_x1_13_inst (.clk(clk_250M), .addr(addr), .q(x1_array[13]));
    rom_x0_14 rom_x0_14_inst (.clk(clk_250M), .addr(addr), .q(x0_array[14]));
    rom_x1_14 rom_x1_14_inst (.clk(clk_250M), .addr(addr), .q(x1_array[14]));
    rom_x0_15 rom_x0_15_inst (.clk(clk_250M), .addr(addr), .q(x0_array[15]));
    rom_x1_15 rom_x1_15_inst (.clk(clk_250M), .addr(addr), .q(x1_array[15]));
    rom_x0_16 rom_x0_16_inst (.clk(clk_250M), .addr(addr), .q(x0_array[16]));
    rom_x1_16 rom_x1_16_inst (.clk(clk_250M), .addr(addr), .q(x1_array[16]));
    rom_x0_17 rom_x0_17_inst (.clk(clk_250M), .addr(addr), .q(x0_array[17]));
    rom_x1_17 rom_x1_17_inst (.clk(clk_250M), .addr(addr), .q(x1_array[17]));
    rom_x0_18 rom_x0_18_inst (.clk(clk_250M), .addr(addr), .q(x0_array[18]));
    rom_x1_18 rom_x1_18_inst (.clk(clk_250M), .addr(addr), .q(x1_array[18]));
    rom_x0_19 rom_x0_19_inst (.clk(clk_250M), .addr(addr), .q(x0_array[19]));
    rom_x1_19 rom_x1_19_inst (.clk(clk_250M), .addr(addr), .q(x1_array[19]));
    rom_x0_20 rom_x0_20_inst (.clk(clk_250M), .addr(addr), .q(x0_array[20]));
    rom_x1_20 rom_x1_20_inst (.clk(clk_250M), .addr(addr), .q(x1_array[20]));
    rom_x0_21 rom_x0_21_inst (.clk(clk_250M), .addr(addr), .q(x0_array[21]));
    rom_x1_21 rom_x1_21_inst (.clk(clk_250M), .addr(addr), .q(x1_array[21]));
    rom_x0_22 rom_x0_22_inst (.clk(clk_250M), .addr(addr), .q(x0_array[22]));
    rom_x1_22 rom_x1_22_inst (.clk(clk_250M), .addr(addr), .q(x1_array[22]));
    rom_x0_23 rom_x0_23_inst (.clk(clk_250M), .addr(addr), .q(x0_array[23]));
    rom_x1_23 rom_x1_23_inst (.clk(clk_250M), .addr(addr), .q(x1_array[23]));
    rom_x0_24 rom_x0_24_inst (.clk(clk_250M), .addr(addr), .q(x0_array[24]));
    rom_x1_24 rom_x1_24_inst (.clk(clk_250M), .addr(addr), .q(x1_array[24]));
    rom_x0_25 rom_x0_25_inst (.clk(clk_250M), .addr(addr), .q(x0_array[25]));
    rom_x1_25 rom_x1_25_inst (.clk(clk_250M), .addr(addr), .q(x1_array[25]));
    rom_x0_26 rom_x0_26_inst (.clk(clk_250M), .addr(addr), .q(x0_array[26]));
    rom_x1_26 rom_x1_26_inst (.clk(clk_250M), .addr(addr), .q(x1_array[26]));
    rom_x0_27 rom_x0_27_inst (.clk(clk_250M), .addr(addr), .q(x0_array[27]));
    rom_x1_27 rom_x1_27_inst (.clk(clk_250M), .addr(addr), .q(x1_array[27]));
    rom_x0_28 rom_x0_28_inst (.clk(clk_250M), .addr(addr), .q(x0_array[28]));
    rom_x1_28 rom_x1_28_inst (.clk(clk_250M), .addr(addr), .q(x1_array[28]));
    rom_x0_29 rom_x0_29_inst (.clk(clk_250M), .addr(addr), .q(x0_array[29]));
    rom_x1_29 rom_x1_29_inst (.clk(clk_250M), .addr(addr), .q(x1_array[29]));

    ram_out ram_out_inst (.clk(clk_250M), .wr_en(wr_en), .ram_addr(ram_addr), .data_in(data_in), .data_out(label_out));

    // svm core instantiations
    genvar i;
    generate
        for (i = 0; i < 30; i++) begin : svm_loop
            svm svm_inst (
                .clk(clk_250M),
                .rst(!rst), //ACTIVE LOW
                .x0(x0_array[i]),
                .x1(x1_array[i]),
                .label(label_array[i])
            );

            assign data_in[i] = label_array[i]; //concatenate and write outputs
        end
    endgenerate

    always_ff @(posedge clk_250M) begin
        if (!rst) begin //ACTIVE LOW
            addr <= 5'b11111;
            ready <= 1'b1;
            done <= 1'b0;
            state <= IDLE;
            wr_en <= 1'b0;
            ram_addr <= 5'b00000;
        end
        else begin
		    // done <= (data_in == 30'h3fff8000) ? 1'b1 : 1'b0 | done; //self checking + sticky bit
            case(state)
                // await start button (ACTIVE LOW)
                IDLE: begin
                        wr_en <= 1'b0;
                        ram_addr <= 5'b00000;
                        if (!start && ready) begin
                            addr <= 5'b00000;
                            ready <= 1'b0;
                            state <= STARTING;
                            done <= 1'b0;
                        end
                        else begin
                            addr <= 5'b11111;
                            ready <= 1'b1;
                            state <= IDLE;
                            done <= done;
                        end
                    end
                
                // read data from rom (1 cycle)
                STARTING: begin
                        wr_en <= 1'b0;
                        ram_addr <= 5'b00000;
                        done <= 1'b0;
                        addr <= 5'b11111;
                        ready <= 1'b0;
                        // state <= PROCESSING;
                        state <= PROCESSING1;
                    end

                // compute and write data into ram (1 cycle)
                // PROCESSING: begin
                //         wr_en <= 1'b1;
                //         ram_addr <= 5'b00000;
                //         done <= 1'b1;
                //         addr <= 5'b11111;
                //         ready <= 1'b1;
                //         state <= IDLE;
                //     end

                // compute and write data into ram (3 cycle)
                PROCESSING1: begin
                        wr_en <= 1'b0;
                        ram_addr <= 5'b00000;
                        done <= 1'b0;
                        addr <= 5'b11111;
                        ready <= 1'b0;
                        state <= PROCESSING2;
                    end

                PROCESSING2: begin
                        wr_en <= 1'b0;
                        ram_addr <= 5'b00000;
                        done <= 1'b0;
                        addr <= 5'b11111;
                        ready <= 1'b0;
                        state <= PROCESSING3;
                    end

                PROCESSING3: begin
                        wr_en <= 1'b1;
                        ram_addr <= 5'b00000;
                        done <= 1'b1;
                        addr <= 5'b11111;
                        ready <= 1'b1;
                        state <= IDLE;
                    end
            endcase
        end
    end

endmodule