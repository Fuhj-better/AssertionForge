// uart_sva.sv
// This file contains SystemVerilog Assertions (SVA) for the UART system.
// It uses a 'checker' module and 'bind' statement to connect assertions
// to the design signals without modifying the RTL.

// Define constants from uart_parser for use in assertions
// These are copied from uart_parser.sv to make the checker self-contained
`define CHAR_CR         8'h0d
`define CHAR_LF         8'h0a
`define CHAR_SPACE      8'h20
`define CHAR_TAB        8'h09
`define CHAR_COMMA      8'h2C
`define CHAR_R_UP       8'h52
`define CHAR_r_LO       8'h72
`define CHAR_W_UP       8'h57
`define CHAR_w_LO       8'h77
`define CHAR_0          8'h30
`define CHAR_1          8'h31
`define CHAR_2          8'h32
`define CHAR_3          8'h33
`define CHAR_4          8'h34
`define CHAR_5          8'h35
`define CHAR_6          8'h36
`define CHAR_7          8'h37
`define CHAR_8          8'h38
`define CHAR_9          8'h39
`define CHAR_A_UP       8'h41
`define CHAR_B_UP       8'h42
`define CHAR_C_UP       8'h43
`define CHAR_D_UP       8'h44
`define CHAR_E_UP       8'h45
`define CHAR_F_UP       8'h46
`define CHAR_a_LO       8'h61
`define CHAR_b_LO       8'h62
`define CHAR_c_LO       8'h63
`define CHAR_d_LO       8'h64
`define CHAR_e_LO       8'h65
`define CHAR_f_LO       8'h66

// main (receive) state machine states
`define MAIN_IDLE       4'b0000
`define MAIN_WHITE1     4'b0001
`define MAIN_DATA       4'b0010
`define MAIN_WHITE2     4'b0011
`define MAIN_ADDR       4'b0100
`define MAIN_EOL        4'b0101
// binary mode extension states
`define MAIN_BIN_CMD    4'b1000
`define MAIN_BIN_ADRH   4'b1001
`define MAIN_BIN_ADRL   4'b1010
`define MAIN_BIN_LEN    4'b1011
`define MAIN_BIN_DATA   4'b1100

// transmit state machine
`define TX_IDLE         3'b000
`define TX_HI_NIB       3'b001
`define TX_LO_NIB       3'b100
`define TX_CHAR_CR      3'b101
`define TX_CHAR_LF      3'b110

// binary extension mode commands - the command is indicated by bits 5:4 of the command byte
`define BIN_CMD_NOP     2'b00
`define BIN_CMD_READ    2'b01
`define BIN_CMD_WRITE   2'b10


// 修改点1: 将 AW 声明为 checker 的参数
module uart_system_assertions #(parameter int AW = 16) (
    input logic clock,
    input logic reset,
    input logic ser_in,
    input logic ser_out,
    input logic [AW-1:0] int_address, // 使用 AW 参数
    input logic [7:0] int_wr_data,
    input logic int_write,
    input logic [7:0] int_rd_data,
    input logic int_read,
    input logic int_req,
    input logic int_gnt,
    input logic tx_busy,
    input logic [7:0] rx_data,
    input logic new_rx_data,
    // Signals from uart_parser1 instance (internal signals)
    input logic [3:0] main_sm,
    input logic data_in_hex_range,
    input logic write_op,
    input logic read_op,
    input logic bin_write_op,
    input logic bin_read_op,
    input logic addr_auto_inc,
    input logic send_stat_flag,
    input logic [7:0] bin_byte_count,
    input logic bin_last_byte,
    input logic tx_end_p,
    input logic [7:0] data_param,
    input logic [15:0] addr_param, // AW is 16 for uart_parser1
    input logic read_done,
    input logic read_done_s,
    input logic [7:0] read_data_s,
    input logic [2:0] tx_sm,
    input logic s_tx_busy,
    input logic [3:0] tx_nibble,
    input logic [7:0] tx_char,
    input logic [3:0] data_nibble,
    input logic write_req, // Internal signal of uart_parser
    // Signals from uart1.uart_rx_1 instance (internal signals)
    input logic [7:0] rx_data_buf, // Renamed from data_buf to avoid conflict with rx_data port
    input logic rx_busy,
    input logic ce_1_mid,
    // Signals from uart1 instance (internal wire)
    input logic ce_16
);

    // 修改点2: 移除此行，因为 AW 已作为 checker 的参数声明
    // localparam AW = 16; // uart_parser is instantiated with #(16)

    // Assertions from property_goldmine.sva, modified with hierarchical paths
    // Note: Some assertions referring to undefined signals (e.g., framing_error)
    // or undefined functions (e.g., expected_data) are commented out.

    assert_a0: assert property(@(posedge clock) int_write |-> $stable(int_wr_data));
    assert_a1: assert property(@(posedge clock) $rose(int_write) |=> $stable(int_address)[*2]);
    assert_a2: assert property(@(posedge clock) $rose(int_gnt) |-> ##[1:5] $fell(int_req));
    assert_a3: assert property(@(posedge clock) tx_busy |-> $changed(ser_out) throughout (1'b1)[*1]);
    assert_a4: assert property(@(posedge clock) $rose(int_req) |-> ##[1:2] $rose(int_gnt));
    assert_a5: assert property(@(posedge clock) $rose(int_write) |=> $fell(int_write));
    assert_a6: assert property(@(posedge clock) $fell(reset) |-> ##[1:5] $stable(ser_out));
    assert_a7: assert property(@(posedge clock) $rose(int_read) |=> !$isunknown(int_rd_data));
    assert_a8: assert property(@(posedge clock) reset |=> (ser_out == 0 && int_address == 0 && int_wr_data == 0 && int_write == 0 && int_read == 0));
    assert_a9: assert property(@(posedge clock) disable iff (reset) $changed(ser_in) |-> ser_in == $past(ser_in));
    assert_a10: assert property(@(posedge clock) int_write |-> ##1 !int_write);
    assert_a11: assert property(@(posedge clock) int_read |-> ##1 !int_read);
    assert_a12: assert property(@(posedge clock) $rose(int_read) |-> ##[1:3] $stable(int_rd_data));
    assert_a13: assert property(@(posedge clock) reset |-> ser_out == 0 [*2] ##1 ser_out == 0);
    assert_a14: assert property(@(posedge clock) $rose(ser_in) |-> ##[1:3] $stable(int_rd_data));
    assert_a15: assert property(@(posedge clock) $fell(tx_busy) |-> ser_out ##1 ser_out ##1 ser_out);
    // assert_a16: assert property(@(posedge clock) $rose(int_write) |-> $past(int_wr_data, 1) == int_wr_data ##1 $future(int_wr_data, 1) == int_wr_data);
    assert_a17: assert property(@(posedge clock) $changed(int_req) |-> $past(int_req) == int_req);
    assert_a18: assert property(@(posedge clock) $rose(int_req) |-> $stable(int_address)[*2]);
    assert_a19: assert property(@(posedge clock) $rose(int_read) |=> $fell(int_read));
    assert_a20: assert property(@(posedge clock) $rose(int_req) |-> ##[1:4] $rose(int_gnt));
    assert_a21: assert property(@(posedge clock) ($rose(int_write) || $rose(int_read)) |=> $stable(int_address));
    assert_a22: assert property(@(posedge clock) tx_busy |-> $stable(ser_out) or $rose(ser_out) or $fell(ser_out));
    assert_a23: assert property(@(posedge clock) $rose(int_write) |=> $stable(int_wr_data));
    assert_a24: assert property(@(posedge clock) disable iff (reset) (main_sm == `MAIN_ADDR && !new_rx_data) |=> $stable(int_address));
    assert_a25: assert property(@(posedge clock) $rose(!reset) |=> $stable(int_address) until_with ((main_sm == `MAIN_ADDR && new_rx_data && !data_in_hex_range) || (main_sm == `MAIN_BIN_LEN && new_rx_data) || (addr_auto_inc && ((bin_read_op && tx_end_p && !bin_last_byte) || (bin_write_op && int_write)))));
    assert_a26: assert property(@(posedge clock) disable iff (reset) !( (main_sm == `MAIN_ADDR && new_rx_data && !data_in_hex_range) || (main_sm == `MAIN_BIN_LEN && new_rx_data) || (addr_auto_inc && ((bin_read_op && tx_end_p && !bin_last_byte) || (bin_write_op && int_write))) ) |=> $stable(int_address));
    assert_a27: assert property(@(posedge clock) disable iff (reset) (addr_auto_inc && bin_read_op && tx_end_p && !bin_last_byte) |=> (int_address == $past(int_address) + 1));
    assert_a28: assert property(@(posedge clock) reset |=> (int_address == 0));
    assert_a29: assert property(@(posedge clock) disable iff (reset) (main_sm == `MAIN_ADDR && new_rx_data && !data_in_hex_range) |=> (int_address == $past(addr_param[AW-1:0])));
    assert_a30: assert property(@(posedge clock) disable iff (reset) (main_sm == `MAIN_BIN_LEN && new_rx_data) |=> (int_address == $past(addr_param[AW-1:0])));
    assert_a31: assert property(@(posedge clock) disable iff (reset) (main_sm == `MAIN_ADDR && new_rx_data && data_in_hex_range) |=> (int_address == $past(int_address)));
    assert_a32: assert property(@(posedge clock) disable iff (reset) !new_rx_data |=> (int_address == $past(int_address)));
    assert_a33: assert property(@(posedge clock) disable iff (reset) (bin_read_op && !addr_auto_inc && tx_end_p) |=> (int_address == $past(int_address)));
    assert_a34: assert property(@(posedge clock) disable iff (reset) (addr_auto_inc && bin_write_op && int_write) |=> (int_address == $past(int_address) + 1));
    assert_a35: assert property(@(posedge clock) disable iff (reset) (main_sm != `MAIN_ADDR && main_sm != `MAIN_BIN_LEN && new_rx_data) |=> (int_address == $past(int_address)));
    assert_a36: assert property(@(posedge clock) disable iff (reset) ((main_sm == `MAIN_BIN_LEN) && new_rx_data) |=> (int_address == $past(addr_param[AW-1:0])));
    assert_a37: assert property(@(posedge clock) $rose(reset) |=> (int_address == 0));
    assert_a38: assert property(@(posedge clock) disable iff (reset) (bin_write_op && !addr_auto_inc) |=> $stable(int_address));
    assert_a39: assert property(@(posedge clock) reset |-> (int_address == 0));
    assert_a40: assert property(@(posedge clock) disable iff (reset) !(reset || ((main_sm == `MAIN_ADDR) && new_rx_data && !data_in_hex_range) || ((main_sm == `MAIN_BIN_LEN) && new_rx_data)) |=> $stable(int_address));
    assert_a41: assert property(@(posedge clock) ((main_sm == `MAIN_BIN_LEN) && new_rx_data) |=> (int_address == addr_param[AW-1:0]));
    assert_a42: assert property(@(posedge clock) $fell(int_read || int_write) |-> !int_gnt);
    assert_a43: assert property(@(posedge clock) $stable(int_req) |-> $stable(int_gnt));
    assert_a44: assert property(@(posedge clock) $fell(int_req) |=> !int_gnt);
    assert_a45: assert property(@(posedge clock) disable iff (reset) $rose(int_req) |-> ##[1:2] $rose(int_gnt));
    assert_a46: assert property(@(posedge clock) (int_gnt && (int_read || int_write)) |-> ##[1:2] int_gnt);
    assert_a47: assert property(@(posedge clock) (int_gnt && int_write && $changed(int_wr_data)) |-> ##[1:2] $stable(int_gnt));
    assert_a48: assert property(@(posedge clock) (int_wr_data == 0) && !int_req |-> !int_gnt);
    assert_a49: assert property(@(posedge clock) disable iff (reset) int_req |-> ##[1:3] int_gnt);
    assert_a50: assert property(@(posedge clock) int_req |-> int_gnt[*1:8]);
    assert_a51: assert property(@(posedge clock) $changed(int_address) |-> !int_gnt);
    assert_a52: assert property(@(posedge clock) int_gnt |-> (int_req && (int_read || int_write)));
    assert_a53: assert property(@(posedge clock) !(reset && int_gnt));
    assert_a54: assert property(@(posedge clock) $fell(reset) |=> !int_gnt);
    assert_a55: assert property(@(posedge clock) reset |-> !int_gnt);
    assert_a56: assert property(@(posedge clock) disable iff (reset) (int_req && !reset) |-> ##[1:4] int_gnt);
    assert_a57: assert property(@(posedge clock) $fell(int_req) |-> ##[0:1] !int_gnt);
    assert_a58: assert property(@(posedge clock) (int_gnt && $changed(int_address)) |=> int_gnt);
    assert_a59: assert property(@(posedge clock) disable iff (reset) (int_req && !reset) |-> ##[1:3] int_gnt);
    assert_a60: assert property(@(posedge clock) disable iff (!reset) !(reset && int_gnt));
    assert_a61: assert property(@(posedge clock) int_gnt |-> (int_write || int_read));
    assert_a62: assert property(@(posedge clock) disable iff (reset) $fell(int_req) |-> ##1 !int_gnt);
    assert_a63: assert property(@(posedge clock) disable iff (reset) int_gnt |-> ##[0:7] !int_gnt);
    assert_a64: assert property(@(posedge clock) disable iff (reset) ($rose(int_req) ##1 int_req[*4]) |-> (int_gnt [*1:5]));
    assert_a65: assert property(@(posedge clock) disable iff (reset) (int_gnt && int_read) |-> ##[1:3] $stable(int_rd_data));
    assert_a66: assert property(@(posedge clock) disable iff (reset) $fell(int_req) |-> ##[1:2] $fell(int_gnt));
    assert_a67: assert property(@(posedge clock) $rose(reset) |-> ##1 !int_gnt);
    assert_a68: assert property(@(posedge clock) disable iff (reset) (int_read && int_gnt) |-> int_gnt throughout int_read[->1]);
    assert_a69: assert property(@(posedge clock) disable iff (reset) (int_read && int_gnt) |-> (int_rd_data >= 8'h00 && int_rd_data <= 8'hFF));
    assert_a70: assert property(@(posedge clock) disable iff (reset) (int_read && int_gnt && (int_rd_data == 0)) |=> (int_rd_data != 0));
    // assert_a71: assert property(@(posedge clock) disable iff (reset) $fell(reset) |-> (int_rd_data == 0) until (int_read && int_gnt));
    assert_a72: assert property(@(posedge clock) disable iff (reset) $fell(reset) |-> ##3 (int_rd_data >= 0 && int_rd_data <= 63));
    assert_a73: assert property(@(posedge clock) disable iff (reset) !int_gnt |-> $stable(int_rd_data));
    assert_a74: assert property(@(posedge clock) disable iff (reset) $changed(int_rd_data) |-> (int_read && int_gnt));
    assert_a75: assert property(@(posedge clock) disable iff (reset) $fell(int_gnt) |-> $stable(int_rd_data)[*1]);
    assert_a76: assert property(@(posedge clock) disable iff (reset) (int_gnt && $past(int_read) && int_read) |-> !$isunknown(int_rd_data));
    assert_a77: assert property(@(posedge clock) disable iff (reset) (int_read && int_gnt) |-> $changed(int_rd_data));
    assert_a78: assert property(@(posedge clock) disable iff (reset) $changed(int_rd_data) |-> $past(int_gnt));
    assert_a79: assert property(@(posedge clock) disable iff (reset) int_read && int_gnt |=> $stable(int_rd_data)[*1]);
    assert_a80: assert property(@(posedge clock) disable iff (reset) !int_read |-> $stable(int_rd_data));
    assert_a81: assert property(@(posedge clock) disable iff (reset) int_read && !int_gnt |-> $stable(int_rd_data));
    assert_a82: assert property(@(posedge clock) disable iff (reset) (!int_read || !int_gnt) && ($changed(ser_in) || $changed(ser_out)) |-> $stable(int_rd_data));
    assert_a83: assert property(@(posedge clock) reset |-> (int_rd_data == 0));
    assert_a84: assert property(@(posedge clock) disable iff (reset) $fell(int_read && int_gnt) |-> $stable(int_rd_data)[*1]);
    assert_a85: assert property(@(posedge clock) disable iff (reset) ($rose(int_read) && int_gnt) |=> $stable(int_rd_data)[*1]);
    assert_a86: assert property(@(posedge clock) reset |-> $stable(int_rd_data));
    assert_a87: assert property(@(posedge clock) disable iff (reset) int_write |-> $stable(int_rd_data));
    assert_a88: assert property(@(posedge clock) disable iff (reset) !int_gnt |-> int_rd_data === 8'bz);
    assert_a89: assert property(@(posedge clock) disable iff (reset) !int_read && $changed(int_address) |-> $stable(int_rd_data));
    assert_a90: assert property(@(posedge clock) disable iff (reset) $changed(int_rd_data) |-> $past(int_read && int_gnt));
    assert_a91: assert property(@(posedge clock) $rose(!reset) |-> (int_rd_data == 0));
    assert_a92: assert property(@(posedge clock) disable iff (reset) $fell(int_read) |-> $stable(int_rd_data)[*1]);
    assert_a93: assert property(@(posedge clock) disable iff (reset) $fell(int_gnt) |-> $stable(int_rd_data)[*2]);
    assert_a94: assert property(@(posedge clock) disable iff (reset) (!int_read || !int_gnt) && $changed(int_address) |-> $stable(int_rd_data));
    assert_a95: assert property(@(posedge clock) disable iff (reset) $fell(int_req) && int_gnt |-> $stable(int_rd_data));
    assert_a96: assert property(@(posedge clock) disable iff (reset) (int_read && int_gnt) |-> ##[1:2] $changed(int_rd_data));
    assert_a97: assert property(@(posedge clock) disable iff (reset) !reset |-> (int_rd_data >= 0 && int_rd_data <= 255));
    assert_a98: assert property(@(posedge clock) reset |-> ##1 (int_rd_data == 0));
    assert_a99: assert property(@(posedge clock) (int_read && int_gnt) |-> ##[1:2] $changed(int_rd_data));
    assert_a100: assert property(@(posedge clock) $changed(int_rd_data) |-> (int_read && int_gnt && !reset));
    assert_a101: assert property(@(posedge clock) $rose(reset) |-> $stable(int_rd_data));
    assert_a102: assert property(@(posedge clock) (!int_read && !int_write) |-> int_rd_data === 'z);
    assert_a103: assert property(@(posedge clock) !(int_read && int_gnt) |-> int_rd_data === 'z);
    assert_a104: assert property(@(posedge clock) int_rd_data !== 'z |-> (int_gnt && int_read));
    assert_a105: assert property(@(posedge clock) (int_read && int_gnt) |=> (int_rd_data >= 0 && int_rd_data <= 255));
    assert_a106: assert property(@(posedge clock) $changed(int_write) && !int_read |-> $stable(int_rd_data));
    assert_a107: assert property(@(posedge clock) (int_rd_data >= 128 && int_rd_data <= 255) |=> (int_rd_data >= 64 && int_rd_data <= 255));
    assert_a108: assert property(@(posedge clock) int_write |-> $stable(int_rd_data));
    assert_a109: assert property(@(posedge clock) $fell(int_gnt) && int_read |=> $stable(int_rd_data));
    assert_a110: assert property(@(posedge clock) (int_read && int_gnt) && $changed(ser_out) |-> $stable(int_rd_data));
    assert_a111: assert property(@(posedge clock) $fell(int_read) |-> $past(int_rd_data) === int_rd_data);
    assert_a112: assert property(@(posedge clock) int_read && $fell(int_req) && !int_gnt |-> $stable(int_rd_data));
    assert_a113: assert property(@(posedge clock) ($rose(int_gnt) && int_read) |-> ##[1:3] (int_rd_data != 0));
    assert_a114: assert property(@(posedge clock) $rose(!reset) |-> (int_rd_data === 8'b0));
    assert_a115: assert property(@(posedge clock) int_read && !int_gnt |-> $stable(int_rd_data));
    assert_a116: assert property(@(posedge clock) $changed(ser_out) |-> $stable(int_rd_data));
    assert_a117: assert property(@(posedge clock) int_gnt && int_read |-> $stable(int_rd_data));
    assert_a118: assert property(@(posedge clock) int_gnt && $changed(int_read) |-> $stable(int_rd_data));
    assert_a119: assert property(@(posedge clock) (int_read && int_gnt && $fell(int_req)) |-> $changed(int_rd_data) [->1]);
    assert_a120: assert property(@(posedge clock) (int_rd_data !== 'z) <-> (int_read && int_gnt));
    assert_a121: assert property(@(posedge clock) (int_read && int_gnt) |-> $changed(int_rd_data) [->1]);
    assert_a122: assert property(@(posedge clock) $changed(int_wr_data) && !int_read |-> $stable(int_rd_data));
    assert_a123: assert property(@(posedge clock) (int_read && int_gnt) && $changed(int_address) |-> $stable(int_rd_data));
    assert_a124: assert property(@(posedge clock) (int_read && int_gnt) && $changed(int_wr_data) |-> $stable(int_rd_data));
    assert_a125: assert property(@(posedge clock) $changed(ser_in) && !(int_read && int_gnt) |-> $stable(int_rd_data));
    assert_a126: assert property(@(posedge clock) disable iff (reset) $rose(int_write) && !int_read |-> $stable(int_rd_data));
    assert_a127: assert property(@(posedge clock) disable iff (reset) (int_req && !int_gnt) |=> $stable(int_rd_data));
    assert_a128: assert property(@(posedge clock) reset |-> !$isunknown(int_rd_data) && int_rd_data == 0);
    assert_a129: assert property(@(posedge clock) disable iff (reset) $fell(int_gnt) |=> $stable(int_rd_data));
    assert_a130: assert property(@(posedge clock) disable iff (reset) int_req && !int_gnt |-> $stable(int_rd_data));
    assert_a131: assert property(@(posedge clock) disable iff (reset) $changed(int_req) && !int_gnt |-> $stable(int_rd_data));
    assert_a132: assert property(@(posedge clock) disable iff (reset) $fell(int_gnt) && $past(int_read) |-> $stable(int_rd_data));
    assert_a133: assert property(@(posedge clock) disable iff (reset) !int_gnt |-> !$isunknown(int_rd_data) && $stable(int_rd_data));
    assert_a134: assert property(@(posedge clock) disable iff (reset) $changed(int_wr_data) && !int_read |-> $stable(int_rd_data));
    assert_a135: assert property(@(posedge clock) disable iff (reset) !int_read && !int_gnt |-> $stable(int_rd_data));
    assert_a136: assert property(@(posedge clock) disable iff (reset) $fell(int_read) && int_gnt |-> $stable(int_rd_data));
    assert_a137: assert property(@(posedge clock) disable iff (reset) $changed(int_wr_data) && !int_write |-> $stable(int_rd_data));
    assert_a138: assert property(@(posedge clock) disable iff (reset) !int_read && !int_gnt && $changed(ser_out) |-> $stable(int_rd_data));
    assert_a139: assert property(@(posedge clock) disable iff (reset) $changed(ser_in) && !(int_read && int_gnt) |-> $stable(int_rd_data));
    assert_a140: assert property(@(posedge clock) disable iff (reset) !$past(int_read) |-> int_rd_data == 0);
    assert_a141: assert property(@(posedge clock) disable iff (reset) $fell(int_read) |=> $stable(int_rd_data));
    assert_a142: assert property(@(posedge clock) disable iff (reset) int_read && int_gnt && $changed(ser_out) |-> $stable(int_rd_data));
    assert_a143: assert property(@(posedge clock) disable iff (reset) (int_rd_data >= 0 && int_rd_data <= 127) |=> (int_rd_data >= 0 && int_rd_data <= 255));
    // assert_a144: assert property(@(posedge clock) disable iff (reset) (int_read && int_gnt) |-> int_rd_data == expected_data(int_address));
    assert_a145: assert property(@(posedge clock) disable iff (reset) int_gnt && $fell(int_read) |-> $stable(int_rd_data));
    assert_a146: assert property(@(posedge clock) disable iff (reset) $rose(int_read && int_gnt) |=> $changed(int_rd_data));
    assert_a147: assert property(@(posedge clock) disable iff (reset) $fell(int_read) |-> ##[0:2] $stable(int_rd_data));
    assert_a148: assert property(@(posedge clock) disable iff (reset) $changed(int_address) && !(int_read && int_gnt) |-> $stable(int_rd_data));
    assert_a149: assert property(@(posedge clock) disable iff (reset) int_read && int_gnt |=> $stable(int_rd_data)[*1:$] until $fell(int_read));
    assert_a150: assert property(@(posedge clock) $fell(reset) |-> ##2 (int_rd_data >= 0 && int_rd_data <= 255));
    assert_a151: assert property(@(posedge clock) reset |=> int_rd_data == 0);
    assert_a152: assert property(@(posedge clock) disable iff (reset) $changed(int_req) && !int_gnt |=> $stable(int_rd_data));
    assert_a153: assert property(@(posedge clock) disable iff (reset) int_req && !int_gnt |-> !$isunknown(int_rd_data));
    assert_a154: assert property(@(posedge clock) disable iff (reset) !int_gnt |-> !$isunknown(int_rd_data));
    assert_a155: assert property(@(posedge clock) disable iff (reset) $rose(int_read && int_gnt) |-> ##[1:2] $changed(int_rd_data));
    assert_a156: assert property(@(posedge clock) reset |-> int_rd_data == 0);
    assert_a157: assert property(@(posedge clock) (int_read && !int_req) |=> !int_read);
    assert_a158: assert property(@(posedge clock) int_read |=> !int_read);
    assert_a159: assert property(@(posedge clock) (int_read && int_req && int_gnt) |=> !int_read);
    assert_a160: assert property(@(posedge clock) int_read |=> $past(int_rd_data,1) == $past(int_rd_data,2));
    assert_a161: assert property(@(posedge clock) int_read |-> !$isunknown(int_address));
    assert_a162: assert property(@(posedge clock) !int_req |-> !int_read);
    assert_a163: assert property(@(posedge clock) int_read |-> $past(int_address,1) == int_address);
    assert_a164: assert property(@(posedge clock) int_read |=> $stable(int_rd_data));
    assert_a165: assert property(@(posedge clock) $stable(int_gnt) && int_gnt |-> !int_read);
    assert_a166: assert property(@(posedge clock) (int_read && int_req && int_gnt) |=> !int_read);
    assert_a167: assert property(@(posedge clock) (int_read && int_req && int_gnt) |=> !int_read throughout !(int_req && int_gnt)[->1]);
    assert_a168: assert property(@(posedge clock) reset |-> !int_read);
    assert_a169: assert property(@(posedge clock) !(int_read && int_write));
    assert_a170: assert property(@(posedge clock) int_read |-> $stable(int_address));
    assert_a171: assert property(@(posedge clock) (int_read && $stable(int_address)) |=> !(int_read && $stable(int_address)));
    assert_a172: assert property(@(posedge clock) int_read |-> (int_req && int_gnt));
    assert_a173: assert property(@(posedge clock) $rose(int_read) |-> $stable(int_address)[*1]);
    assert_a174: assert property(@(posedge clock) $rose(int_read) |-> (int_req throughout ( ##[0:$] $fell(int_req) )));
    assert_a175: assert property(@(posedge clock) $stable(int_gnt) && int_gnt |-> !int_read);
    assert_a176: assert property(@(posedge clock) $rose(int_read) |-> $past(int_req));
    assert_a177: assert property(@(posedge clock) $rose(int_read) |-> (int_req && int_gnt));
    assert_a178: assert property(@(posedge clock) !(int_read && int_write));
    assert_a179: assert property(@(posedge clock) reset |-> !int_read);
    assert_a180: assert property(@(posedge clock) int_read |-> int_gnt);
    assert_a181: assert property(@(posedge clock) int_read |-> (int_req && int_gnt));
    assert_a182: assert property(@(posedge clock) disable iff (reset) ($rose(int_req) && !int_gnt [*5]) |-> ##1 !int_req);
    assert_a183: assert property(@(posedge clock) disable iff (reset) (int_req && (int_write || int_read)) |-> !int_gnt until !(int_write || int_read));
    assert_a184: assert property(@(posedge clock) (int_req && (int_read || int_write)) |-> int_gnt);
    assert_a185: assert property(@(posedge clock) $rose(int_req) |-> ##1 int_req);
    assert_a186: assert property(@(posedge clock) int_req |-> !int_gnt);
    assert_a187: assert property(@(posedge clock) disable iff (reset) ($fell(int_gnt)) |-> ##[0:2] !int_req);
    assert_a188: assert property(@(posedge clock) disable iff (reset) (int_req && !int_gnt) |-> ##1 !int_req);
    assert_a189: assert property(@(posedge clock) disable iff (reset) (int_req && int_gnt) |-> (int_rd_data == int_address || int_wr_data == int_address));
    assert_a190: assert property(@(posedge clock) !(int_req && int_write));
    assert_a191: assert property(@(posedge clock) int_req |-> (int_read || int_write));
    assert_a192: assert property(@(posedge clock) disable iff (reset) int_rd_data |-> !int_req);
    assert_a193: assert property(@(posedge clock) disable iff (reset) (int_rd_data || int_wr_data) |-> !int_req);
    assert_a194: assert property(@(posedge clock) $rose(reset) |-> !int_req);
    assert_a195: assert property(@(posedge clock) disable iff (reset) int_req |-> (int_address != 0));
    assert_a196: assert property(@(posedge clock) $rose(!reset) |=> !int_req[*2]);
    assert_a197: assert property(@(posedge clock) disable iff (reset) int_req |-> ##[1:3] int_gnt);
    assert_a198: assert property(@(posedge clock) disable iff (reset) ((int_rd_data && !int_read) || (int_wr_data && !int_write)) |-> !int_req);
    assert_a199: assert property(@(posedge clock) reset |-> !int_req);
    // assert_a200: assert property(@(posedge clock) disable iff (reset) $rose(int_req) |-> ($stable(int_address) throughout (int_req until int_gnt)));
    assert_a201: assert property(@(posedge clock) $fell(reset) |-> ##2 !int_req);
    assert_a202: assert property(@(posedge clock) disable iff (reset) $fell(int_req) |-> ##1 !int_gnt);
    assert_a203: assert property(@(posedge clock) disable iff (reset) (int_req && ser_in) |-> ##[1:2] int_gnt);
    assert_a204: assert property(@(posedge clock) $rose(reset) |-> ##1 !int_req);
    assert_a205: assert property(@(posedge clock) disable iff (reset) (int_req && int_read) |-> ##[1:3] int_gnt);
    assert_a206: assert property(@(posedge clock) disable iff (reset) int_req |=> (int_req throughout int_gnt[->1]) or reset);
    assert_a207: assert property(@(posedge clock) disable iff (reset) (int_req && !int_gnt) |-> $stable(int_wr_data) throughout (int_req[*1:$] ##0 int_gnt));
    assert_a208: assert property(@(posedge clock) disable iff (reset) !write_req && !reset |=> $stable(int_wr_data));
    assert_a209: assert property(@(posedge clock) disable iff (reset) $changed(int_wr_data) |=> ##1 $stable(int_wr_data) or reset);
    assert_a210: assert property(@(posedge clock) disable iff (reset) (write_op || bin_write_op) && !new_rx_data |=> $stable(int_wr_data));
    assert_a211: assert property(@(posedge clock) disable iff (reset) (write_op && (main_sm == `MAIN_ADDR) && new_rx_data && !data_in_hex_range) |=> (int_wr_data == $past(data_param)));
    assert_a212: assert property(@(posedge clock) disable iff (reset) write_op && !data_in_hex_range && !bin_write_op |=> (int_wr_data == $past(data_param)));
    assert_a213: assert property(@(posedge clock) reset |-> (int_wr_data == 0) [*1:$] until !reset);
    assert_a214: assert property(@(posedge clock) disable iff (reset) (bin_write_op && (main_sm == `MAIN_BIN_DATA) && new_rx_data) |=> (int_wr_data == $past(rx_data)));
    assert_a215: assert property(@(posedge clock) disable iff (reset) $changed(int_wr_data) |-> ##[1:$] $stable(int_wr_data)[*1]);
    assert_a216: assert property(@(posedge clock) disable iff (reset) new_rx_data && !(main_sm == `MAIN_ADDR || main_sm == `MAIN_BIN_DATA) |=> $stable(int_wr_data));
    assert_a217: assert property(@(posedge clock) disable iff (reset) (write_op || bin_write_op) |=> $changed(int_wr_data));
    assert_a218: assert property(@(posedge clock) disable iff (reset) write_op && !new_rx_data |=> $stable(int_wr_data));
    assert_a219: assert property(@(posedge clock) disable iff (reset) !(write_op || bin_write_op) && (main_sm == `MAIN_ADDR || main_sm == `MAIN_BIN_DATA) |=> $stable(int_wr_data));
    assert_a220: assert property(@(posedge clock) disable iff (reset) ($rose(write_op)) ##1 (write_op && (main_sm == `MAIN_ADDR) && new_rx_data && !data_in_hex_range) |=> (int_wr_data == $past(data_param)));
    assert_a221: assert property(@(posedge clock) reset |=> (int_wr_data == 0) and (!reset throughout (write_op || bin_write_op)[->1]) |=> $stable(int_wr_data));
    assert_a222: assert property(@(posedge clock) disable iff (reset) !reset && !(write_op && (main_sm == `MAIN_ADDR) && new_rx_data && !data_in_hex_range) && !(bin_write_op && (main_sm == `MAIN_BIN_DATA) && new_rx_data) |=> ($stable(int_wr_data)));
    assert_a223: assert property(@(posedge clock) disable iff (reset) ($changed(int_wr_data)) |=> ($stable(int_wr_data)[*1] or (write_op && (main_sm == `MAIN_ADDR) && new_rx_data && !data_in_hex_range) or (bin_write_op && (main_sm == `MAIN_BIN_DATA) && new_rx_data)));
    assert_a224: assert property(@(posedge clock) disable iff (reset) (write_op && (main_sm != `MAIN_ADDR)) |=> ($stable(int_wr_data)));
    assert_a225: assert property(@(posedge clock) disable iff (reset) ($fell(write_req)) |-> ##[1:2] ($stable(int_wr_data)));
    assert_a226: assert property(@(posedge clock) disable iff (reset) ($changed(write_op) || $changed(bin_write_op)) |-> $stable(int_wr_data));
    assert_a227: assert property(@(posedge clock) disable iff (reset) $rose(write_op || bin_write_op) |-> !$isunknown(int_wr_data));
    assert_a228: assert property(@(posedge clock) reset |-> (int_wr_data == 0));
    assert_a229: assert property(@(posedge clock) $changed(int_wr_data) |-> $past(write_op || bin_write_op));
    assert_a230: assert property(@(posedge clock) disable iff (reset) $changed(int_wr_data) |-> ($past(new_rx_data) && ($past(write_op) || $past(bin_write_op))));
    assert_a231: assert property(@(posedge clock) $rose(!reset) |=> (int_wr_data == 0) until $rose(write_op));
    assert_a232: assert property(@(posedge clock) disable iff (reset) $changed(int_wr_data) |-> ($past(write_op && (main_sm == `MAIN_ADDR) && new_rx_data && !data_in_hex_range) || $past(bin_write_op && (main_sm == `MAIN_BIN_DATA) && new_rx_data)));
    assert_a233: assert property(@(posedge clock) (write_op && bin_write_op) |-> if (main_sm == `MAIN_ADDR) (int_wr_data == data_param) else if (main_sm == `MAIN_BIN_DATA) (int_wr_data == rx_data));
    assert_a234: assert property(@(posedge clock) disable iff (reset) !(write_op || bin_write_op) |=> $stable(int_wr_data));
    assert_a235: assert property(@(posedge clock) (write_op && (main_sm == `MAIN_ADDR) && new_rx_data && !data_in_hex_range) |=> (int_wr_data == $past(data_param)));
    assert_a236: assert property(@(posedge clock) reset |=> (int_wr_data == 0));
    assert_a237: assert property(@(posedge clock) disable iff (reset) (bin_write_op && (main_sm != `MAIN_BIN_DATA)) |=> $stable(int_wr_data));
    assert_a238: assert property(@(posedge clock) $fell(reset) |=> (int_wr_data == 0) until (write_op || bin_write_op));
    assert_a239: assert property(@(posedge clock) (int_gnt && write_req) |=> int_write);
    assert_a240: assert property(@(posedge clock) (int_gnt && write_req) |-> (int_write && !write_req));
    assert_a241: assert property(@(posedge clock) (int_gnt && write_req) |=> int_write ##1 !int_write);
    assert_a242: assert property(@(posedge clock) (!write_req && int_gnt) |=> !int_write);
    assert_a243: assert property(@(posedge clock) (int_write && !(int_gnt && write_req)) |=> !int_write);
    assert_a244: assert property(@(posedge clock) $stable(int_gnt) && $stable(write_req) |-> $stable(int_write));
    assert_a245: assert property(@(posedge clock) (!int_gnt && !write_req) |=> !int_write);
    assert_a246: assert property(@(posedge clock) reset |=> !int_write);
    assert_a247: assert property(@(posedge clock) (int_gnt && write_req) |=> (int_write && !write_req));
    assert_a248: assert property(@(posedge clock) (int_write && !(int_gnt && write_req)) |=> !int_write);
    assert_a249: assert property(@(posedge clock) !write_req |-> !int_write);
    assert_a250: assert property(@(posedge clock) $rose(int_gnt) && !write_req |-> !int_write);
    assert_a251: assert property(@(posedge clock) $fell(int_gnt) && write_req |=> !int_write);
    assert_a252: assert property(@(posedge clock) int_write |-> int_gnt && write_req);
    assert_a253: assert property(@(posedge clock) int_write |=> (int_gnt && write_req) || !int_write);
    assert_a254: assert property(@(posedge clock) int_gnt && !write_req |=> !int_write);
    assert_a255: assert property(@(posedge clock) reset |-> !int_write);
    assert_a256: assert property(@(posedge clock) write_req && !int_gnt |=> !int_write);
    assert_a257: assert property(@(posedge clock) disable iff (reset) int_write == (int_gnt && write_req));
    assert_a258: assert property(@(posedge clock) disable iff (reset) (!int_gnt || !write_req) |=> !int_write);
    assert_a259: assert property(@(posedge clock) $rose(!reset) |=> (int_write == (int_gnt && write_req)));
    assert_a260: assert property(@(posedge clock) disable iff (reset) (int_gnt && write_req) |=> (int_write && !write_req));
    assert_a261: assert property(@(posedge clock) disable iff (reset) (!int_gnt && write_req) |=> !int_write);
    assert_a262: assert property(@(posedge clock) disable iff (reset) (bin_write_op && (main_sm == `MAIN_BIN_DATA) && new_rx_data) |=> write_req);
    assert_a263: assert property(@(posedge clock) (reset && int_gnt && write_req) |=> !int_write);
    assert_a264: assert property(@(posedge clock) $rose(reset) |=> (rx_data == 8'h0));
    assert_a265: assert property(@(posedge clock) $fell(reset) |=> $stable(ser_out)[*2]);
    assert_a266: assert property(@(posedge clock) $fell(reset) ##1 (rx_busy && ce_1_mid) |=> (rx_data_buf != $past(rx_data_buf)));
    assert_a267: assert property(@(posedge clock) reset |-> !int_gnt);
    assert_a268: assert property(@(posedge clock) reset throughout (##[1:$] !reset) |-> $stable(rx_data));
    assert_a269: assert property(@(posedge clock) $rose(reset) |=> (!int_write && !int_read));
    assert_a270: assert property(@(posedge clock) $rose(reset) |=> !tx_busy);
    assert_a271: assert property(@(posedge clock) $rose(reset) |=> (new_rx_data == 1'b0));
    assert_a272: assert property(@(posedge clock) reset |-> !(rx_busy && ce_1_mid && !reset));
    assert_a273: assert property(@(posedge clock) $rose(reset) |=> (int_rd_data == '0));
    assert_a274: assert property(@(posedge clock) $rose(reset) |=> (int_address == '0));
    // assert_a275: assert property(@(posedge clock) disable iff (reset) !new_rx_data throughout (1'b1)[*0:$] until (valid_reception_condition));
    // assert_a276: assert property(@(posedge clock) $rose(reset) |-> (rx_data == 8'h0 && new_rx_data == 1'b0 && rx_data_buf == 8'h0) regardless_of (ser_in));
    // assert_a277: assert property(@(posedge clock) $fell(reset) |=> $stable(rx_data) until (rx_data_change_condition));
    assert_a278: assert property(@(posedge clock) $rose(reset) |=> (rx_data == 8'h0 && new_rx_data == 1'b0 && rx_data_buf == 8'h0));
    assert_a279: assert property(@(posedge clock) $rose(reset) |=> !int_req);
    assert_a280: assert property(@(posedge clock) $fell(reset) |-> ##[1:3] $changed(ser_in));
    assert_a281: assert property(@(posedge clock) reset |-> (!int_write && !int_read));
    assert_a282: assert property(@(posedge clock) $rose(reset) |=> (rx_data_buf == 8'h0));
    assert_a283: assert property(@(posedge clock) $rose(reset) |-> reset[*2]);
    assert_a284: assert property(@(posedge clock) $fell(reset) |=> $stable(rx_data) && $stable(rx_data_buf));
    assert_a285: assert property(@(posedge clock) $rose(reset) |=> !new_rx_data);
    assert_a286: assert property(@(posedge clock) $rose(reset) |=> (int_rd_data == 8'h0));
    assert_a287: assert property(@(posedge clock) $rose(reset) |=> $stable(ser_out) throughout (reset[*1:$]));
    // assert_a288: assert property(@(posedge clock) disable iff (reset) $rose(ser_in) && $fell(ser_in) within [0:$] |-> $stable(ser_out));
    assert_a289: assert property(@(posedge clock) disable iff (reset) (!tx_busy) |-> ##4 (ser_out == $past(ser_in,4)));
    assert_a290: assert property(@(posedge clock) disable iff (reset || !ce_16) ($fell(ser_in) ##1 !$rose(ser_in)[*10]) |-> !new_rx_data);
    assert_a291: assert property(@(posedge clock) disable iff (reset) $fell(ser_in) |-> ##[1:16] int_req);
    // assert_a292: assert property(@(posedge clock) disable iff (reset) ($rose(ser_in) && $fell(ser_in) within [0:$]) |-> $stable(int_wr_data));
    assert_a293: assert property(@(posedge clock) disable iff (reset) (ser_in == 0)[*16] |-> $stable(int_address));
    // assert_a294: assert property(@(posedge clock) disable iff (reset) ($stable(ser_in)[*8] && int_req) |-> ##1 int_gnt);
    assert_a295: assert property(@(posedge clock) (reset) |-> $stable(ser_in));
    assert_a296: assert property(@(posedge clock) disable iff (reset) ($fell(ser_in) ##1 $changed(ser_in)[*8]) |-> ##[1:16] int_write);
    // assert_a297: assert property(@(posedge ce_16) $rose(ser_in) && $fell(ser_in) within [0:$] |-> $stable(ser_in));
    // assert_a298: assert property(@(posedge clock) $rose(!reset) |-> $stable(ser_in, 2));
    assert_a299: assert property(@(posedge clock) reset |-> (ser_out == $past(ser_out,1)) and (!$changed(int_write)) and (!$changed(int_read)));
    assert_a300: assert property(@(posedge clock) disable iff (reset) (int_read && $fell(ser_in)) |=> (int_read throughout !$fell(ser_in))[*1:$] ##1 $fell(ser_in));
    assert_a301: assert property(@(posedge clock) disable iff (reset) ser_in[*10] |-> !int_read);
    assert_a302: assert property(@(posedge clock) disable iff (reset) $fell(ser_in) |-> ##[1:3] int_req);
    // assert_a303: assert property(@(posedge clock) disable iff (reset) (ser_in == 0) throughout (ce_16[->24]) |-> framing_error);
    // assert_a304: assert property(@(posedge clock) disable iff (reset) $rose(!reset) ##4 (!reset throughout (($stable(ser_in, 2)) or ($fell(ser_in) ##2 $stable(ser_in)))));
    assert_a305: assert property(@(posedge clock) disable iff (reset) $fell(ser_in) ##[1:$] $rose(ser_in) |-> ##[0:16] int_write [*1] ##1 !int_write);
    // assert_a306: assert property(@(posedge clock) ($fell(reset) && (ser_in == 0)) ##1 (ser_in == 1)[*4] |-> !framing_error);
    // assert_a307: assert property(@(posedge clock) disable iff (reset) (frame_end && ce_16) |-> ##[1:2] new_rx_data);
    assert_a308: assert property(@(posedge clock) $rose(!reset) |-> strong(##[0:16] $stable(ser_in)));
    assert_a309: assert property(@(posedge clock) disable iff (reset) ($fell(ser_in) ##1 ser_in[*8]) |-> ##[1:24] (int_wr_data == $past(ser_in, 8)));
    assert_a310: assert property(@(posedge clock) disable iff (reset) $fell(ser_in) && ce_16 |-> ##[1:16] $rose(ser_in));
    assert_a311: assert property(@(posedge clock) disable iff (reset) $fell(ser_in) |-> ##[1:16] (new_rx_data && $stable(ser_out)));
    assert_a312: assert property(@(posedge clock) disable iff (reset) ($countones(ser_in) >= 10) |-> $stable(ser_in) until $fell(ser_in));
    assert_a313: assert property(@(posedge clock) disable iff (reset) ($countones(ser_in) >= 16) |-> ##[1:16] !int_read);
    // assert_a314: assert property(@(posedge clock) disable iff (reset) (ser_in == 1'b0 ##1 ser_in == 1'b1 ##1 ser_in == 1'b0 ##1 ser_in == 1'b1 ##1 ser_in == 1'b0 ##1 ser_in == 1'b1 ##1 ser_in == 1'b0 ##1 ser_in == 1'b1) && ce_16 |-> (int_address == 8'h55));
    assert_a315: assert property(@(posedge clock) reset |-> $stable(int_rd_data));
    assert_a316: assert property(@(posedge clock) disable iff (reset) ($fell(ser_in) && ce_16) |-> ##[1:$] (new_rx_data && $stable(ce_16)[*1:$]));
    assert_a317: assert property(@(posedge clock) disable iff (reset) ($rose(ser_in) && int_req) |-> ##[1:2] !int_gnt);
    assert_a318: assert property(@(posedge clock) $rose(int_gnt) |-> ##[1:2] $stable(ser_out));
    assert_a319: assert property(@(posedge clock) int_write |-> ##[1:2] (ser_out == 0 || ser_out == 1));
    assert_a320: assert property(@(posedge clock) int_gnt && !int_req |-> $stable(ser_out));
    assert_a321: assert property(@(posedge clock) (int_req && !int_gnt) |-> ##5 ($stable(ser_out)[*5]));
    assert_a322: assert property(@(posedge clock) (int_req && !int_gnt) |-> ##[1:10] $changed(ser_out));
    assert_a323: assert property(@(posedge clock) reset |-> $stable(ser_out));
    assert_a324: assert property(@(posedge clock) (int_read && int_write) |-> $stable(ser_out));
    assert_a325: assert property(@(posedge clock) $fell(reset) |-> (ser_out == 0)[*3]);
    assert_a326: assert property(@(posedge clock) (!int_read && !int_write) |-> ##4 ($fell(ser_out) || $rose(ser_out))[*4]);
    assert_a327: assert property(@(posedge clock) $changed(int_address) && !int_write |-> $stable(ser_out));
    assert_a328: assert property(@(posedge clock) $changed(int_read) || $changed(int_write) |-> ##1 !$isunknown(ser_out) throughout ##[0:1] $stable(ser_out));
    assert_a329: assert property(@(posedge clock) disable iff (reset) $changed(ser_out) |-> $rose(clock));
    assert_a330: assert property(@(posedge clock) (int_req && !int_gnt) |-> $stable(ser_out));
    assert_a331: assert property(@(posedge clock) int_write |-> ser_out == int_wr_data[0]);
    assert_a332: assert property(@(posedge clock) reset |=> $stable(ser_out));
    assert_a333: assert property(@(posedge clock) $fell(reset) |-> (ser_out == 1'b0));
    assert_a334: assert property(@(posedge clock) $fell(int_write) |=> (ser_out == 1'b1));
    assert_a335: assert property(@(posedge clock) ($changed(int_address) && int_write) |-> (ser_out == 1'b1)[->1] ##1 ser_out != 1'b1);
    assert_a336: assert property(@(posedge clock) disable iff (int_write) (ser_in |-> ##1 ser_out == $past(ser_in)));
    assert_a337: assert property(@(posedge clock) (int_req && int_gnt) |-> ##[1:4] !$stable(ser_out));
    assert_a338: assert property(@(posedge clock) $rose(int_read) |-> ##[1:10] $changed(ser_out));
    assert_a339: assert property(@(posedge clock) disable iff (reset) $changed(int_rd_data) |-> ##[1:2] $stable(ser_out));
    assert_a340: assert property(@(posedge clock) disable iff (reset) int_write |-> (ser_out == int_wr_data[0]));
    assert_a341: assert property(@(posedge clock) $fell(reset) |-> ##[1:2] (ser_out == 1'b1));
    assert_a342: assert property(@(posedge clock) $rose(reset) |-> ##[0:2] ($stable(ser_out)) and $fell(reset) |-> ##[0:2] ($stable(ser_out)));
    // assert_a343: assert property(@(posedge clock) disable iff (reset) (int_read && $changed(int_rd_data)) |-> ##[1:4] (ser_out == $past(int_rd_data, 4)[7:0]));
    assert_a344: assert property(@(posedge clock) disable iff (reset) (int_write && int_gnt)[*3] |-> (ser_out == int_wr_data[7:0]));
    assert_a345: assert property(@(posedge clock) disable iff (reset) $fell(int_read) |-> (ser_out == 1'b1)[*3]);
    assert_a346: assert property(@(posedge clock) disable iff (reset) (int_read && int_gnt) |-> ##[1:3] (ser_out == $past(ser_in, 3)));
    assert_a347: assert property(@(posedge clock) disable iff (reset) ($past(int_address) == 8'h00 && int_address != 8'h00) |-> ##[1:5] (ser_out == 1'b1));
    assert_a348: assert property(@(posedge clock) disable iff (reset) int_read |-> (ser_out == ^int_rd_data));

endmodule

// Bind the checker to the uart2bus_top module instance
// This allows the checker to observe signals within uart2bus_top and its sub-modules.
// 修改点3: 在 bind 语句中传递 AW 参数的值
bind uart2bus_top uart_system_assertions #(16) uart_sva_inst (
    .clock(clock),
    .reset(reset),
    .ser_in(ser_in),
    .ser_out(ser_out),
    .int_address(int_address),
    .int_wr_data(int_wr_data),
    .int_write(int_write),
    .int_rd_data(int_rd_data),
    .int_read(int_read),
    .int_req(int_req),
    .int_gnt(int_gnt),
    .tx_busy(tx_busy),
    .rx_data(rx_data),
    .new_rx_data(new_rx_data),
    // Hierarchical paths for internal signals of uart_parser1
    .main_sm(uart_parser1.main_sm),
    .data_in_hex_range(uart_parser1.data_in_hex_range),
    .write_op(uart_parser1.write_op),
    .read_op(uart_parser1.read_op),
    .bin_write_op(uart_parser1.bin_write_op),
    .bin_read_op(uart_parser1.bin_read_op),
    .addr_auto_inc(uart_parser1.addr_auto_inc),
    .send_stat_flag(uart_parser1.send_stat_flag),
    .bin_byte_count(uart_parser1.bin_byte_count),
    .bin_last_byte(uart_parser1.bin_last_byte),
    .tx_end_p(uart_parser1.tx_end_p),
    .data_param(uart_parser1.data_param),
    .addr_param(uart_parser1.addr_param),
    .read_done(uart_parser1.read_done),
    .read_done_s(uart_parser1.read_done_s),
    .read_data_s(uart_parser1.read_data_s),
    .tx_sm(uart_parser1.tx_sm),
    .s_tx_busy(uart_parser1.s_tx_busy),
    .tx_nibble(uart_parser1.tx_nibble),
    .tx_char(uart_parser1.tx_char),
    .data_nibble(uart_parser1.data_nibble),
    .write_req(uart_parser1.write_req),
    // Hierarchical paths for internal signals of uart1.uart_rx_1
    .rx_data_buf(uart1.uart_rx_1.data_buf),
    .rx_busy(uart1.uart_rx_1.rx_busy),
    .ce_1_mid(uart1.uart_rx_1.ce_1_mid),
    // Hierarchical path for internal wire of uart1
    .ce_16(uart1.ce_16)
);
