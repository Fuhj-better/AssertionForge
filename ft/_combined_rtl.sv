
// File: i2c_master_defines.v
/////////////////////////////////////////////////////////////////////
////                                                             ////
////  WISHBONE rev.B2 compliant I2C Master controller defines    ////
////                                                             ////
////                                                             ////
////  Author: Richard Herveille                                  ////
////          richard@asics.ws                                   ////
////          www.asics.ws                                       ////
////                                                             ////
////  Downloaded from: http://www.opencores.org/projects/i2c/    ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2001 Richard Herveille                        ////
////                    richard@asics.ws                         ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////

//  CVS Log
//
//  $Id: i2c_master_defines.v,v 1.3 2001-11-05 11:59:25 rherveille Exp $
//
//  $Date: 2001-11-05 11:59:25 $
//  $Revision: 1.3 $
//  $Author: rherveille $
//  $Locker:  $
//  $State: Exp $
//
// Change History:
//               $Log: not supported by cvs2svn $


// I2C registers wishbone addresses

// bitcontroller states
`define I2C_CMD_NOP   4'b0000
`define I2C_CMD_START 4'b0001
`define I2C_CMD_STOP  4'b0010
`define I2C_CMD_WRITE 4'b0100
`define I2C_CMD_READ  4'b1000

// File: timescale.v
`timescale 1ns / 10ps


// File: i2c_master_bit_ctrl.v
/////////////////////////////////////////////////////////////////////
////                                                             ////
////  WISHBONE rev.B2 compliant I2C Master bit-controller        ////
////                                                             ////
////                                                             ////
////  Author: Richard Herveille                                  ////
////          richard@asics.ws                                   ////
////          www.asics.ws                                       ////
////                                                             ////
////  Downloaded from: http://www.opencores.org/projects/i2c/    ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2001 Richard Herveille                        ////
////                    richard@asics.ws                         ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////

//  CVS Log
//
//  $Id: i2c_master_bit_ctrl.v,v 1.14 2009-01-20 10:25:29 rherveille Exp $
//
//  $Date: 2009-01-20 10:25:29 $
//  $Revision: 1.14 $
//  $Author: rherveille $
//  $Locker:  $
//  $State: Exp $
//
// Change History:
//               $Log: $
//               Revision 1.14  2009/01/20 10:25:29  rherveille
//               Added clock synchronization logic
//               Fixed slave_wait signal
//
//               Revision 1.13  2009/01/19 20:29:26  rherveille
//               Fixed synopsys miss spell (synopsis)
//               Fixed cr[0] register width
//               Fixed ! usage instead of ~
//               Fixed bit controller parameter width to 18bits
//
//               Revision 1.12  2006/09/04 09:08:13  rherveille
//               fixed short scl high pulse after clock stretch
//               fixed slave model not returning correct '(n)ack' signal
//
//               Revision 1.11  2004/05/07 11:02:26  rherveille
//               Fixed a bug where the core would signal an arbitration lost (AL bit set), when another master controls the bus and the other master generates a STOP bit.
//
//               Revision 1.10  2003/08/09 07:01:33  rherveille
//               Fixed a bug in the Arbitration Lost generation caused by delay on the (external) sda line.
//               Fixed a potential bug in the byte controller's host-acknowledge generation.
//
//               Revision 1.9  2003/03/10 14:26:37  rherveille
//               Fixed cmd_ack generation item (no bug).
//
//               Revision 1.8  2003/02/05 00:06:10  rherveille
//               Fixed a bug where the core would trigger an erroneous 'arbitration lost' interrupt after being reset, when the reset pulse width < 3 clk cycles.
//
//               Revision 1.7  2002/12/26 16:05:12  rherveille
//               Small code simplifications
//
//               Revision 1.6  2002/12/26 15:02:32  rherveille
//               Core is now a Multimaster I2C controller
//
//               Revision 1.5  2002/11/30 22:24:40  rherveille
//               Cleaned up code
//
//               Revision 1.4  2002/10/30 18:10:07  rherveille
//               Fixed some reported minor start/stop generation timing issuess.
//
//               Revision 1.3  2002/06/15 07:37:03  rherveille
//               Fixed a small timing bug in the bit controller.\nAdded verilog simulation environment.
//
//               Revision 1.2  2001/11/05 11:59:25  rherveille
//               Fixed wb_ack_o generation bug.
//               Fixed bug in the byte_controller statemachine.
//               Added headers.
//

//
/////////////////////////////////////
// Bit controller section
/////////////////////////////////////
//
// Translate simple commands into SCL/SDA transitions
// Each command has 5 states, A/B/C/D/idle
//
// start:	SCL	~~~~~~~~~~\____
//	SDA	~~~~~~~~\______
//		 x | A | B | C | D | i
//
// repstart	SCL	____/~~~~\___
//	SDA	__/~~~\______
//		 x | A | B | C | D | i
//
// stop	SCL	____/~~~~~~~~
//	SDA	==\____/~~~~~
//		 x | A | B | C | D | i
//
//- write	SCL	____/~~~~\____
//	SDA	==X=========X=
//		 x | A | B | C | D | i
//
//- read	SCL	____/~~~~\____
//	SDA	XXXX=====XXXX
//		 x | A | B | C | D | i
//

// Timing:     Normal mode      Fast mode
///////////////////////////////////////////////////////////////////////
// Fscl        100KHz           400KHz
// Th_scl      4.0us            0.6us   High period of SCL
// Tl_scl      4.7us            1.3us   Low period of SCL
// Tsu:sta     4.7us            0.6us   setup time for a repeated start condition
// Tsu:sto     4.0us            0.6us   setup time for a stop conditon
// Tbuf        4.7us            1.3us   Bus free time between a stop and start condition
//

// synopsys translate_off
// synopsys translate_on

module i2c_master_bit_ctrl (
    input             clk,      // system clock
    input             rst,      // synchronous active high reset
    input             nReset,   // asynchronous active low reset
    input             ena,      // core enable signal

    input      [15:0] clk_cnt,  // clock prescale value

    input      [ 3:0] cmd,      // command (from byte controller)
    output reg        cmd_ack,  // command complete acknowledge
    output reg        busy,     // i2c bus busy
    output reg        al,       // i2c bus arbitration lost

    input             din,
    output reg        dout,

    input             scl_i,    // i2c clock line input
    output            scl_o,    // i2c clock line output
    output reg        scl_oen,  // i2c clock line output enable (active low)
    input             sda_i,    // i2c data line input
    output            sda_o,    // i2c data line output
    output reg        sda_oen   // i2c data line output enable (active low)
);


    //
    // variable declarations
    //

    reg [ 1:0] cSCL, cSDA;      // capture SCL and SDA
    reg [ 2:0] fSCL, fSDA;      // SCL and SDA filter inputs
    reg        sSCL, sSDA;      // filtered and synchronized SCL and SDA inputs
    reg        dSCL, dSDA;      // delayed versions of sSCL and sSDA
    reg        dscl_oen;        // delayed scl_oen
    reg        sda_chk;         // check SDA output (Multi-master arbitration)
    reg        clk_en;          // clock generation signals
    reg        slave_wait;      // slave inserts wait states
    reg [15:0] cnt;             // clock divider counter (synthesis)
    reg [13:0] filter_cnt;      // clock divider for filter


    // state machine variable
    reg [17:0] c_state; // synopsys enum_state

    //
    // module body
    //

    // whenever the slave is not ready it can delay the cycle by pulling SCL low
    // delay scl_oen
    always @(posedge clk)
      dscl_oen <= #1 scl_oen;

    // slave_wait is asserted when master wants to drive SCL high, but the slave pulls it low
    // slave_wait remains asserted until the slave releases SCL
    always @(posedge clk or negedge nReset)
      if (!nReset) slave_wait <= 1'b0;
      else         slave_wait <= (scl_oen & ~dscl_oen & ~sSCL) | (slave_wait & ~sSCL);

    // master drives SCL high, but another master pulls it low
    // master start counting down its low cycle now (clock synchronization)
    wire scl_sync   = dSCL & ~sSCL & scl_oen;


    // generate clk enable signal
    always @(posedge clk or negedge nReset)
      if (~nReset)
      begin
          cnt    <= #1 16'h0;
          clk_en <= #1 1'b1;
      end
      else if (rst || ~|cnt || !ena || scl_sync)
      begin
          cnt    <= #1 clk_cnt;
          clk_en <= #1 1'b1;
      end
      else if (slave_wait)
      begin
          cnt    <= #1 cnt;
          clk_en <= #1 1'b0;    
      end
      else
      begin
          cnt    <= #1 cnt - 16'h1;
          clk_en <= #1 1'b0;
      end


    // generate bus status controller

    // capture SDA and SCL
    // reduce metastability risk
    always @(posedge clk or negedge nReset)
      if (!nReset)
      begin
          cSCL <= #1 2'b00;
          cSDA <= #1 2'b00;
      end
      else if (rst)
      begin
          cSCL <= #1 2'b00;
          cSDA <= #1 2'b00;
      end
      else
      begin
          cSCL <= {cSCL[0],scl_i};
          cSDA <= {cSDA[0],sda_i};
      end


    // filter SCL and SDA signals; (attempt to) remove glitches
    always @(posedge clk or negedge nReset)
      if      (!nReset     ) filter_cnt <= 14'h0;
      else if (rst || !ena ) filter_cnt <= 14'h0;
      else if (~|filter_cnt) filter_cnt <= clk_cnt >> 2; //16x I2C bus frequency
      else                   filter_cnt <= filter_cnt -1;


    always @(posedge clk or negedge nReset)
      if (!nReset)
      begin
          fSCL <= 3'b111;
          fSDA <= 3'b111;
      end
      else if (rst)
      begin
          fSCL <= 3'b111;
          fSDA <= 3'b111;
      end
      else if (~|filter_cnt)
      begin
          fSCL <= {fSCL[1:0],cSCL[1]};
          fSDA <= {fSDA[1:0],cSDA[1]};
      end


    // generate filtered SCL and SDA signals
    always @(posedge clk or negedge nReset)
      if (~nReset)
      begin
          sSCL <= #1 1'b1;
          sSDA <= #1 1'b1;

          dSCL <= #1 1'b1;
          dSDA <= #1 1'b1;
      end
      else if (rst)
      begin
          sSCL <= #1 1'b1;
          sSDA <= #1 1'b1;

          dSCL <= #1 1'b1;
          dSDA <= #1 1'b1;
      end
      else
      begin
          sSCL <= #1 &fSCL[2:1] | &fSCL[1:0] | (fSCL[2] & fSCL[0]);
          sSDA <= #1 &fSDA[2:1] | &fSDA[1:0] | (fSDA[2] & fSDA[0]);

          dSCL <= #1 sSCL;
          dSDA <= #1 sSDA;
      end

    // detect start condition => detect falling edge on SDA while SCL is high
    // detect stop condition => detect rising edge on SDA while SCL is high
    reg sta_condition;
    reg sto_condition;
    always @(posedge clk or negedge nReset)
      if (~nReset)
      begin
          sta_condition <= #1 1'b0;
          sto_condition <= #1 1'b0;
      end
      else if (rst)
      begin
          sta_condition <= #1 1'b0;
          sto_condition <= #1 1'b0;
      end
      else
      begin
          sta_condition <= #1 ~sSDA &  dSDA & sSCL;
          sto_condition <= #1  sSDA & ~dSDA & sSCL;
      end


    // generate i2c bus busy signal
    always @(posedge clk or negedge nReset)
      if      (!nReset) busy <= #1 1'b0;
      else if (rst    ) busy <= #1 1'b0;
      else              busy <= #1 (sta_condition | busy) & ~sto_condition;


    // generate arbitration lost signal
    // aribitration lost when:
    // 1) master drives SDA high, but the i2c bus is low
    // 2) stop detected while not requested
    reg cmd_stop;
    always @(posedge clk or negedge nReset)
      if (~nReset)
          cmd_stop <= #1 1'b0;
      else if (rst)
          cmd_stop <= #1 1'b0;
      else if (clk_en)
          cmd_stop <= #1 cmd == `I2C_CMD_STOP;

    always @(posedge clk or negedge nReset)
      if (~nReset)
          al <= #1 1'b0;
      else if (rst)
          al <= #1 1'b0;
      else
          al <= #1 (sda_chk & ~sSDA & sda_oen) | (|c_state & sto_condition & ~cmd_stop);


    // generate dout signal (store SDA on rising edge of SCL)
    always @(posedge clk)
      if (sSCL & ~dSCL) dout <= #1 sSDA;


    // generate statemachine

    // nxt_state decoder
    parameter [17:0] idle    = 18'b0_0000_0000_0000_0000;
    parameter [17:0] start_a = 18'b0_0000_0000_0000_0001;
    parameter [17:0] start_b = 18'b0_0000_0000_0000_0010;
    parameter [17:0] start_c = 18'b0_0000_0000_0000_0100;
    parameter [17:0] start_d = 18'b0_0000_0000_0000_1000;
    parameter [17:0] start_e = 18'b0_0000_0000_0001_0000;
    parameter [17:0] stop_a  = 18'b0_0000_0000_0010_0000;
    parameter [17:0] stop_b  = 18'b0_0000_0000_0100_0000;
    parameter [17:0] stop_c  = 18'b0_0000_0000_1000_0000;
    parameter [17:0] stop_d  = 18'b0_0000_0001_0000_0000;
    parameter [17:0] rd_a    = 18'b0_0000_0010_0000_0000;
    parameter [17:0] rd_b    = 18'b0_0000_0100_0000_0000;
    parameter [17:0] rd_c    = 18'b0_0000_1000_0000_0000;
    parameter [17:0] rd_d    = 18'b0_0001_0000_0000_0000;
    parameter [17:0] wr_a    = 18'b0_0010_0000_0000_0000;
    parameter [17:0] wr_b    = 18'b0_0100_0000_0000_0000;
    parameter [17:0] wr_c    = 18'b0_1000_0000_0000_0000;
    parameter [17:0] wr_d    = 18'b1_0000_0000_0000_0000;

    always @(posedge clk or negedge nReset)
      if (!nReset)
      begin
          c_state <= #1 idle;
          cmd_ack <= #1 1'b0;
          scl_oen <= #1 1'b1;
          sda_oen <= #1 1'b1;
          sda_chk <= #1 1'b0;
      end
      else if (rst | al)
      begin
          c_state <= #1 idle;
          cmd_ack <= #1 1'b0;
          scl_oen <= #1 1'b1;
          sda_oen <= #1 1'b1;
          sda_chk <= #1 1'b0;
      end
      else
      begin
          cmd_ack   <= #1 1'b0; // default no command acknowledge + assert cmd_ack only 1clk cycle

          if (clk_en)
              case (c_state) // synopsys full_case parallel_case
                    // idle state
                    idle:
                    begin
                        case (cmd) // synopsys full_case parallel_case
                             `I2C_CMD_START: c_state <= #1 start_a;
                             `I2C_CMD_STOP:  c_state <= #1 stop_a;
                             `I2C_CMD_WRITE: c_state <= #1 wr_a;
                             `I2C_CMD_READ:  c_state <= #1 rd_a;
                             default:        c_state <= #1 idle;
                        endcase

                        scl_oen <= #1 scl_oen; // keep SCL in same state
                        sda_oen <= #1 sda_oen; // keep SDA in same state
                        sda_chk <= #1 1'b0;    // don't check SDA output
                    end

                    // start
                    start_a:
                    begin
                        c_state <= #1 start_b;
                        scl_oen <= #1 scl_oen; // keep SCL in same state
                        sda_oen <= #1 1'b1;    // set SDA high
                        sda_chk <= #1 1'b0;    // don't check SDA output
                    end

                    start_b:
                    begin
                        c_state <= #1 start_c;
                        scl_oen <= #1 1'b1; // set SCL high
                        sda_oen <= #1 1'b1; // keep SDA high
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    start_c:
                    begin
                        c_state <= #1 start_d;
                        scl_oen <= #1 1'b1; // keep SCL high
                        sda_oen <= #1 1'b0; // set SDA low
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    start_d:
                    begin
                        c_state <= #1 start_e;
                        scl_oen <= #1 1'b1; // keep SCL high
                        sda_oen <= #1 1'b0; // keep SDA low
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    start_e:
                    begin
                        c_state <= #1 idle;
                        cmd_ack <= #1 1'b1;
                        scl_oen <= #1 1'b0; // set SCL low
                        sda_oen <= #1 1'b0; // keep SDA low
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    // stop
                    stop_a:
                    begin
                        c_state <= #1 stop_b;
                        scl_oen <= #1 1'b0; // keep SCL low
                        sda_oen <= #1 1'b0; // set SDA low
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    stop_b:
                    begin
                        c_state <= #1 stop_c;
                        scl_oen <= #1 1'b1; // set SCL high
                        sda_oen <= #1 1'b0; // keep SDA low
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    stop_c:
                    begin
                        c_state <= #1 stop_d;
                        scl_oen <= #1 1'b1; // keep SCL high
                        sda_oen <= #1 1'b0; // keep SDA low
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    stop_d:
                    begin
                        c_state <= #1 idle;
                        cmd_ack <= #1 1'b1;
                        scl_oen <= #1 1'b1; // keep SCL high
                        sda_oen <= #1 1'b1; // set SDA high
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    // read
                    rd_a:
                    begin
                        c_state <= #1 rd_b;
                        scl_oen <= #1 1'b0; // keep SCL low
                        sda_oen <= #1 1'b1; // tri-state SDA
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    rd_b:
                    begin
                        c_state <= #1 rd_c;
                        scl_oen <= #1 1'b1; // set SCL high
                        sda_oen <= #1 1'b1; // keep SDA tri-stated
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    rd_c:
                    begin
                        c_state <= #1 rd_d;
                        scl_oen <= #1 1'b1; // keep SCL high
                        sda_oen <= #1 1'b1; // keep SDA tri-stated
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    rd_d:
                    begin
                        c_state <= #1 idle;
                        cmd_ack <= #1 1'b1;
                        scl_oen <= #1 1'b0; // set SCL low
                        sda_oen <= #1 1'b1; // keep SDA tri-stated
                        sda_chk <= #1 1'b0; // don't check SDA output
                    end

                    // write
                    wr_a:
                    begin
                        c_state <= #1 wr_b;
                        scl_oen <= #1 1'b0; // keep SCL low
                        sda_oen <= #1 din;  // set SDA
                        sda_chk <= #1 1'b0; // don't check SDA output (SCL low)
                    end

                    wr_b:
                    begin
                        c_state <= #1 wr_c;
                        scl_oen <= #1 1'b1; // set SCL high
                        sda_oen <= #1 din;  // keep SDA
                        sda_chk <= #1 1'b0; // don't check SDA output yet
                                            // allow some time for SDA and SCL to settle
                    end

                    wr_c:
                    begin
                        c_state <= #1 wr_d;
                        scl_oen <= #1 1'b1; // keep SCL high
                        sda_oen <= #1 din;
                        sda_chk <= #1 1'b1; // check SDA output
                    end

                    wr_d:
                    begin
                        c_state <= #1 idle;
                        cmd_ack <= #1 1'b1;
                        scl_oen <= #1 1'b0; // set SCL low
                        sda_oen <= #1 din;
                        sda_chk <= #1 1'b0; // don't check SDA output (SCL low)
                    end

              endcase
      end


    // assign scl and sda output (always gnd)
    assign scl_o = 1'b0;
    assign sda_o = 1'b0;

endmodule

// File: i2c_master_byte_ctrl.v
/////////////////////////////////////////////////////////////////////
////                                                             ////
////  WISHBONE rev.B2 compliant I2C Master byte-controller       ////
////                                                             ////
////                                                             ////
////  Author: Richard Herveille                                  ////
////          richard@asics.ws                                   ////
////          www.asics.ws                                       ////
////                                                             ////
////  Downloaded from: http://www.opencores.org/projects/i2c/    ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2001 Richard Herveille                        ////
////                    richard@asics.ws                         ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////

//  CVS Log
//
//  $Id: i2c_master_byte_ctrl.v,v 1.8 2009-01-19 20:29:26 rherveille Exp $
//
//  $Date: 2009-01-19 20:29:26 $
//  $Revision: 1.8 $
//  $Author: rherveille $
//  $Locker:  $
//  $State: Exp $
//
// Change History:
//               $Log: not supported by cvs2svn $
//               Revision 1.7  2004/02/18 11:40:46  rherveille
//               Fixed a potential bug in the statemachine. During a 'stop' 2 cmd_ack signals were generated. Possibly canceling a new start command.
//
//               Revision 1.6  2003/08/09 07:01:33  rherveille
//               Fixed a bug in the Arbitration Lost generation caused by delay on the (external) sda line.
//               Fixed a potential bug in the byte controller's host-acknowledge generation.
//
//               Revision 1.5  2002/12/26 15:02:32  rherveille
//               Core is now a Multimaster I2C controller
//
//               Revision 1.4  2002/11/30 22:24:40  rherveille
//               Cleaned up code
//
//               Revision 1.3  2001/11/05 11:59:25  rherveille
//               Fixed wb_ack_o generation bug.
//               Fixed bug in the byte_controller statemachine.
//               Added headers.
//

// synopsys translate_off
// synopsys translate_on

module i2c_master_byte_ctrl (
	clk, rst, nReset, ena, clk_cnt, start, stop, read, write, ack_in, din,
	cmd_ack, ack_out, dout, i2c_busy, i2c_al, scl_i, scl_o, scl_oen, sda_i, sda_o, sda_oen );

	//
	// inputs & outputs
	//
	input clk;     // master clock
	input rst;     // synchronous active high reset
	input nReset;  // asynchronous active low reset
	input ena;     // core enable signal

	input [15:0] clk_cnt; // 4x SCL

	// control inputs
	input       start;
	input       stop;
	input       read;
	input       write;
	input       ack_in;
	input [7:0] din;

	// status outputs
	output       cmd_ack;
	reg cmd_ack;
	output       ack_out;
	reg ack_out;
	output       i2c_busy;
	output       i2c_al;
	output [7:0] dout;

	// I2C signals
	input  scl_i;
	output scl_o;
	output scl_oen;
	input  sda_i;
	output sda_o;
	output sda_oen;


	//
	// Variable declarations
	//

	// statemachine
	parameter [4:0] ST_IDLE  = 5'b0_0000;
	parameter [4:0] ST_START = 5'b0_0001;
	parameter [4:0] ST_READ  = 5'b0_0010;
	parameter [4:0] ST_WRITE = 5'b0_0100;
	parameter [4:0] ST_ACK   = 5'b0_1000;
	parameter [4:0] ST_STOP  = 5'b1_0000;

	// signals for bit_controller
	reg  [3:0] core_cmd;
	reg        core_txd;
	wire       core_ack, core_rxd;

	// signals for shift register
	reg [7:0] sr; //8bit shift register
	reg       shift, ld;

	// signals for state machine
	wire       go;
	reg  [2:0] dcnt;
	wire       cnt_done;

	//
	// Module body
	//

	// hookup bit_controller
	i2c_master_bit_ctrl bit_controller (
		.clk     ( clk      ),
		.rst     ( rst      ),
		.nReset  ( nReset   ),
		.ena     ( ena      ),
		.clk_cnt ( clk_cnt  ),
		.cmd     ( core_cmd ),
		.cmd_ack ( core_ack ),
		.busy    ( i2c_busy ),
		.al      ( i2c_al   ),
		.din     ( core_txd ),
		.dout    ( core_rxd ),
		.scl_i   ( scl_i    ),
		.scl_o   ( scl_o    ),
		.scl_oen ( scl_oen  ),
		.sda_i   ( sda_i    ),
		.sda_o   ( sda_o    ),
		.sda_oen ( sda_oen  )
	);

	// generate go-signal
	assign go = (read | write | stop) & ~cmd_ack;

	// assign dout output to shift-register
	assign dout = sr;

	// generate shift register
	always @(posedge clk or negedge nReset)
	  if (!nReset)
	    sr <= #1 8'h0;
	  else if (rst)
	    sr <= #1 8'h0;
	  else if (ld)
	    sr <= #1 din;
	  else if (shift)
	    sr <= #1 {sr[6:0], core_rxd};

	// generate counter
	always @(posedge clk or negedge nReset)
	  if (!nReset)
	    dcnt <= #1 3'h0;
	  else if (rst)
	    dcnt <= #1 3'h0;
	  else if (ld)
	    dcnt <= #1 3'h7;
	  else if (shift)
	    dcnt <= #1 dcnt - 3'h1;

	assign cnt_done = ~(|dcnt);

	//
	// state machine
	//
	reg [4:0] c_state; // synopsys enum_state

	always @(posedge clk or negedge nReset)
	  if (!nReset)
	    begin
	        core_cmd <= #1 `I2C_CMD_NOP;
	        core_txd <= #1 1'b0;
	        shift    <= #1 1'b0;
	        ld       <= #1 1'b0;
	        cmd_ack  <= #1 1'b0;
	        c_state  <= #1 ST_IDLE;
	        ack_out  <= #1 1'b0;
	    end
	  else if (rst | i2c_al)
	   begin
	       core_cmd <= #1 `I2C_CMD_NOP;
	       core_txd <= #1 1'b0;
	       shift    <= #1 1'b0;
	       ld       <= #1 1'b0;
	       cmd_ack  <= #1 1'b0;
	       c_state  <= #1 ST_IDLE;
	       ack_out  <= #1 1'b0;
	   end
	else
	  begin
	      // initially reset all signals
	      core_txd <= #1 sr[7];
	      shift    <= #1 1'b0;
	      ld       <= #1 1'b0;
	      cmd_ack  <= #1 1'b0;

	      case (c_state) // synopsys full_case parallel_case
	        ST_IDLE:
	          if (go)
	            begin
	                if (start)
	                  begin
	                      c_state  <= #1 ST_START;
	                      core_cmd <= #1 `I2C_CMD_START;
	                  end
	                else if (read)
	                  begin
	                      c_state  <= #1 ST_READ;
	                      core_cmd <= #1 `I2C_CMD_READ;
	                  end
	                else if (write)
	                  begin
	                      c_state  <= #1 ST_WRITE;
	                      core_cmd <= #1 `I2C_CMD_WRITE;
	                  end
	                else // stop
	                  begin
	                      c_state  <= #1 ST_STOP;
	                      core_cmd <= #1 `I2C_CMD_STOP;
	                  end

	                ld <= #1 1'b1;
	            end

	        ST_START:
	          if (core_ack)
	            begin
	                if (read)
	                  begin
	                      c_state  <= #1 ST_READ;
	                      core_cmd <= #1 `I2C_CMD_READ;
	                  end
	                else
	                  begin
	                      c_state  <= #1 ST_WRITE;
	                      core_cmd <= #1 `I2C_CMD_WRITE;
	                  end

	                ld <= #1 1'b1;
	            end

	        ST_WRITE:
	          if (core_ack)
	            if (cnt_done)
	              begin
	                  c_state  <= #1 ST_ACK;
	                  core_cmd <= #1 `I2C_CMD_READ;
	              end
	            else
	              begin
	                  c_state  <= #1 ST_WRITE;       // stay in same state
	                  core_cmd <= #1 `I2C_CMD_WRITE; // write next bit
	                  shift    <= #1 1'b1;
	              end

	        ST_READ:
	          if (core_ack)
	            begin
	                if (cnt_done)
	                  begin
	                      c_state  <= #1 ST_ACK;
	                      core_cmd <= #1 `I2C_CMD_WRITE;
	                  end
	                else
	                  begin
	                      c_state  <= #1 ST_READ;       // stay in same state
	                      core_cmd <= #1 `I2C_CMD_READ; // read next bit
	                  end

	                shift    <= #1 1'b1;
	                core_txd <= #1 ack_in;
	            end

	        ST_ACK:
	          if (core_ack)
	            begin
	               if (stop)
	                 begin
	                     c_state  <= #1 ST_STOP;
	                     core_cmd <= #1 `I2C_CMD_STOP;
	                 end
	               else
	                 begin
	                     c_state  <= #1 ST_IDLE;
	                     core_cmd <= #1 `I2C_CMD_NOP;

	                     // generate command acknowledge signal
	                     cmd_ack  <= #1 1'b1;
	                 end

	                 // assign ack_out output to bit_controller_rxd (contains last received bit)
	                 ack_out <= #1 core_rxd;

	                 core_txd <= #1 1'b1;
	             end
	           else
	             core_txd <= #1 ack_in;

	        ST_STOP:
	          if (core_ack)
	            begin
	                c_state  <= #1 ST_IDLE;
	                core_cmd <= #1 `I2C_CMD_NOP;

	                // generate command acknowledge signal
	                cmd_ack  <= #1 1'b1;
	            end

	      endcase
	  end
endmodule

// File: i2c_master_top.v
/////////////////////////////////////////////////////////////////////
////                                                             ////
////  WISHBONE revB.2 compliant I2C Master controller Top-level  ////
////                                                             ////
////                                                             ////
////  Author: Richard Herveille                                  ////
////          richard@asics.ws                                   ////
////          www.asics.ws                                       ////
////                                                             ////
////  Downloaded from: http://www.opencores.org/projects/i2c/    ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2001 Richard Herveille                        ////
////                    richard@asics.ws                         ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////

//  CVS Log
//
//  $Id: i2c_master_top.v,v 1.12 2009-01-19 20:29:26 rherveille Exp $
//
//  $Date: 2009-01-19 20:29:26 $
//  $Revision: 1.12 $
//  $Author: rherveille $
//  $Locker:  $
//  $State: Exp $
//
// Change History:
//               Revision 1.11  2005/02/27 09:26:24  rherveille
//               Fixed register overwrite issue.
//               Removed full_case pragma, replaced it by a default statement.
//
//               Revision 1.10  2003/09/01 10:34:38  rherveille
//               Fix a blocking vs. non-blocking error in the wb_dat output mux.
//
//               Revision 1.9  2003/01/09 16:44:45  rherveille
//               Fixed a bug in the Command Register declaration.
//
//               Revision 1.8  2002/12/26 16:05:12  rherveille
//               Small code simplifications
//
//               Revision 1.7  2002/12/26 15:02:32  rherveille
//               Core is now a Multimaster I2C controller
//
//               Revision 1.6  2002/11/30 22:24:40  rherveille
//               Cleaned up code
//
//               Revision 1.5  2001/11/10 10:52:55  rherveille
//               Changed PRER reset value from 0x0000 to 0xffff, conform specs.
//

// synopsys translate_off
// synopsys translate_on

module i2c_master_top(
	wb_clk_i, wb_rst_i, arst_i, wb_adr_i, wb_dat_i, wb_dat_o,
	wb_we_i, wb_stb_i, wb_cyc_i, wb_ack_o, wb_inta_o,
	scl_pad_i, scl_pad_o, scl_padoen_o, sda_pad_i, sda_pad_o, sda_padoen_o );

	// parameters
	parameter ARST_LVL = 1'b0; // asynchronous reset level

	//
	// inputs & outputs
	//

	// wishbone signals
	input        wb_clk_i;     // master clock input
	input        wb_rst_i;     // synchronous active high reset
	input        arst_i;       // asynchronous reset
	input  [2:0] wb_adr_i;     // lower address bits
	input  [7:0] wb_dat_i;     // databus input
	output [7:0] wb_dat_o;     // databus output
	input        wb_we_i;      // write enable input
	input        wb_stb_i;     // stobe/core select signal
	input        wb_cyc_i;     // valid bus cycle input
	output       wb_ack_o;     // bus cycle acknowledge output
	output       wb_inta_o;    // interrupt request signal output

	reg [7:0] wb_dat_o;
	reg wb_ack_o;
	reg wb_inta_o;

	// I2C signals
	// i2c clock line
	input  scl_pad_i;       // SCL-line input
	output scl_pad_o;       // SCL-line output (always 1'b0)
	output scl_padoen_o;    // SCL-line output enable (active low)

	// i2c data line
	input  sda_pad_i;       // SDA-line input
	output sda_pad_o;       // SDA-line output (always 1'b0)
	output sda_padoen_o;    // SDA-line output enable (active low)


	//
	// variable declarations
	//

	// registers
	reg  [15:0] prer; // clock prescale register
	reg  [ 7:0] ctr;  // control register
	reg  [ 7:0] txr;  // transmit register
	wire [ 7:0] rxr;  // receive register
	reg  [ 7:0] cr;   // command register
	wire [ 7:0] sr;   // status register

	// done signal: command completed, clear command register
	wire done;

	// core enable signal
	wire core_en;
	wire ien;

	// status register signals
	wire irxack;
	reg  rxack;       // received aknowledge from slave
	reg  tip;         // transfer in progress
	reg  irq_flag;    // interrupt pending flag
	wire i2c_busy;    // bus busy (start signal detected)
	wire i2c_al;      // i2c bus arbitration lost
	reg  al;          // status register arbitration lost bit

	//
	// module body
	//

	// generate internal reset
	wire rst_i = arst_i ^ ARST_LVL;

	// generate wishbone signals
	wire wb_wacc = wb_we_i & wb_ack_o;

	// generate acknowledge output signal
	always @(posedge wb_clk_i)
	  wb_ack_o <= #1 wb_cyc_i & wb_stb_i & ~wb_ack_o; // because timing is always honored

	// assign DAT_O
	always @(posedge wb_clk_i)
	begin
	  case (wb_adr_i) // synopsys parallel_case
	    3'b000: wb_dat_o <= #1 prer[ 7:0];
	    3'b001: wb_dat_o <= #1 prer[15:8];
	    3'b010: wb_dat_o <= #1 ctr;
	    3'b011: wb_dat_o <= #1 rxr; // write is transmit register (txr)
	    3'b100: wb_dat_o <= #1 sr;  // write is command register (cr)
	    3'b101: wb_dat_o <= #1 txr;
	    3'b110: wb_dat_o <= #1 cr;
	    3'b111: wb_dat_o <= #1 0;   // reserved
	  endcase
	end

	// generate registers
	always @(posedge wb_clk_i or negedge rst_i)
	  if (!rst_i)
	    begin
	        prer <= #1 16'hffff;
	        ctr  <= #1  8'h0;
	        txr  <= #1  8'h0;
	    end
	  else if (wb_rst_i)
	    begin
	        prer <= #1 16'hffff;
	        ctr  <= #1  8'h0;
	        txr  <= #1  8'h0;
	    end
	  else
	    if (wb_wacc)
	      case (wb_adr_i) // synopsys parallel_case
	         3'b000 : prer [ 7:0] <= #1 wb_dat_i;
	         3'b001 : prer [15:8] <= #1 wb_dat_i;
	         3'b010 : ctr         <= #1 wb_dat_i;
	         3'b011 : txr         <= #1 wb_dat_i;
	         default: ;
	      endcase

	// generate command register (special case)
	always @(posedge wb_clk_i or negedge rst_i)
	  if (!rst_i)
	    cr <= #1 8'h0;
	  else if (wb_rst_i)
	    cr <= #1 8'h0;
	  else if (wb_wacc)
	    begin
	        if (core_en & (wb_adr_i == 3'b100) )
	          cr <= #1 wb_dat_i;
	    end
	  else
	    begin
	        if (done | i2c_al)
	          cr[7:4] <= #1 4'h0;           // clear command bits when done
	                                        // or when aribitration lost
	        cr[2:1] <= #1 2'b0;             // reserved bits
	        cr[0]   <= #1 1'b0;             // clear IRQ_ACK bit
	    end


	// decode command register
	wire sta  = cr[7];
	wire sto  = cr[6];
	wire rd   = cr[5];
	wire wr   = cr[4];
	wire ack  = cr[3];
	wire iack = cr[0];

	// decode control register
	assign core_en = ctr[7];
	assign ien = ctr[6];

	// hookup byte controller block
	i2c_master_byte_ctrl byte_controller (
		.clk      ( wb_clk_i     ),
		.rst      ( wb_rst_i     ),
		.nReset   ( rst_i        ),
		.ena      ( core_en      ),
		.clk_cnt  ( prer         ),
		.start    ( sta          ),
		.stop     ( sto          ),
		.read     ( rd           ),
		.write    ( wr           ),
		.ack_in   ( ack          ),
		.din      ( txr          ),
		.cmd_ack  ( done         ),
		.ack_out  ( irxack       ),
		.dout     ( rxr          ),
		.i2c_busy ( i2c_busy     ),
		.i2c_al   ( i2c_al       ),
		.scl_i    ( scl_pad_i    ),
		.scl_o    ( scl_pad_o    ),
		.scl_oen  ( scl_padoen_o ),
		.sda_i    ( sda_pad_i    ),
		.sda_o    ( sda_pad_o    ),
		.sda_oen  ( sda_padoen_o )
	);

	// status register block + interrupt request signal
	always @(posedge wb_clk_i or negedge rst_i)
	  if (!rst_i)
	    begin
	        al       <= #1 1'b0;
	        rxack    <= #1 1'b0;
	        tip      <= #1 1'b0;
	        irq_flag <= #1 1'b0;
	    end
	  else if (wb_rst_i)
	    begin
	        al       <= #1 1'b0;
	        rxack    <= #1 1'b0;
	        tip      <= #1 1'b0;
	        irq_flag <= #1 1'b0;
	    end
	  else
	    begin
	        al       <= #1 i2c_al | (al & ~sta);
	        rxack    <= #1 irxack;
	        tip      <= #1 (rd | wr);
	        irq_flag <= #1 (done | i2c_al | irq_flag) & ~iack; // interrupt request flag is always generated
	    end

	// generate interrupt request signals
	always @(posedge wb_clk_i or negedge rst_i)
	  if (!rst_i)
	    wb_inta_o <= #1 1'b0;
	  else if (wb_rst_i)
	    wb_inta_o <= #1 1'b0;
	  else
	    wb_inta_o <= #1 irq_flag && ien; // interrupt signal is only generated when IEN (interrupt enable bit is set)

	// assign status register bits
	assign sr[7]   = rxack;
	assign sr[6]   = i2c_busy;
	assign sr[5]   = al;
	assign sr[4:2] = 3'h0; // reserved
	assign sr[1]   = tip;
	assign sr[0]   = irq_flag;

assert_a0: assert property(@(posedge wb_clk_i) $fell(arst_i) && !wb_rst_i |=> (wb_ack_o == 1'b0 && wb_dat_o == '0 && wb_inta_o == 1'b0) throughout (arst_i == 1'b0)[*1:$]);
assert_a1: assert property(@(posedge wb_clk_i) $fell(arst_i) && wb_cyc_i && wb_stb_i && !wb_we_i |=> (wb_dat_o == '0) throughout (arst_i == 1'b0)[*1:$]);
assert_a2: assert property(@(posedge wb_clk_i) $fell(arst_i) ##[0:1] $rose(arst_i) |=> (scl_padoen_o == 1'b1 && sda_padoen_o == 1'b1) until $rose(wb_clk_i));
assert_a3: assert property(@(posedge wb_clk_i) $rose(arst_i) |=> (scl_pad_o == 1'b1 && sda_pad_o == 1'b1) until (wb_stb_i && wb_cyc_i && $rose(wb_clk_i)));
assert_a4: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> ##1 (scl_pad_o == 1'b1 && scl_padoen_o == 1'b1 && sda_pad_o == 1'b1 && sda_padoen_o == 1'b1 && wb_ack_o == 1'b0 && wb_dat_o == '0 && wb_inta_o == 1'b0));
assert_a5: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> ##1 (tip == 1'b0 && irq_flag == 1'b0 && al == 1'b0));
assert_a6: assert property(@(posedge wb_clk_i) $rose(arst_i) && wb_we_i |=> (wb_dat_o == $past(wb_dat_i, 1)) within ($fell(arst_i)[->1]));
assert_a7: assert property(@(posedge wb_clk_i) $fell(arst_i) && !wb_rst_i |=> (scl_padoen_o == 1'b1 && sda_padoen_o == 1'b1));
assert_a8: assert property(@(posedge wb_clk_i) (arst_i && tip) |-> ##1 (scl_padoen_o && sda_padoen_o));
assert_a9: assert property(@(posedge wb_clk_i) (arst_i && $past(wb_rst_i,1)) |-> (scl_padoen_o && sda_padoen_o));
assert_a10: assert property(@(posedge wb_clk_i) arst_i |-> ##2 (!wb_ack_o && !wb_inta_o && scl_padoen_o && sda_padoen_o));
assert_a11: assert property(@(posedge wb_clk_i) (arst_i && (scl_pad_o || sda_pad_o)) |-> ##2 (scl_padoen_o && sda_padoen_o));
assert_a12: assert property(@(posedge wb_clk_i) (arst_i && wb_stb_i && wb_cyc_i && !wb_we_i) |-> ##1 (!wb_ack_o && (wb_dat_o == 0)));
assert_a13: assert property(@(posedge wb_clk_i) (arst_i && wb_rst_i) |-> ##1 (!tip && !irq_flag && !al));
assert_a14: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> ##1 (wb_stb_i && wb_cyc_i) |-> ##1 wb_ack_o);
assert_a15: assert property(@(posedge wb_clk_i) (arst_i && wb_rst_i) |-> $past(arst_i));
assert_a16: assert property(@(posedge wb_clk_i) (arst_i) |-> ((arst_i ^ ARST_LVL) ? (scl_padoen_o && sda_padoen_o) : (!scl_padoen_o || !sda_padoen_o)));
assert_a17: assert property(@(posedge wb_clk_i) !(arst_i && wb_rst_i));
assert_a18: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> ##[1:2] (wb_cyc_i && wb_stb_i));
assert_a19: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> ##[0:3] (!i2c_busy && !tip && !irq_flag && !al));
assert_a20: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> (scl_padoen_o && sda_padoen_o) throughout (##[0:$] (wb_cyc_i && wb_stb_i)[->1]));
assert_a21: assert property(@(posedge wb_clk_i) arst_i |-> (!wb_ack_o && !wb_inta_o));
assert_a22: assert property(@(posedge wb_clk_i) (arst_i || wb_rst_i) |-> (wb_dat_o == '0 && !wb_inta_o));
assert_a23: assert property(@(posedge wb_clk_i) not (arst_i && wb_rst_i));
assert_a24: assert property(@(posedge wb_clk_i) (arst_i && wb_rst_i) |-> !wb_ack_o until $fell(arst_i));
assert_a25: assert property(@(posedge wb_clk_i) $rose(arst_i) |-> ##[1:2] (!wb_ack_o && wb_dat_o == '0 && !wb_inta_o && !scl_pad_o && scl_padoen_o && !sda_pad_o && sda_padoen_o));
assert_a26: assert property(@(posedge wb_clk_i) $rose(arst_i) && i2c_busy |-> ##[1:2] (al && scl_padoen_o && sda_padoen_o));
assert_a27: assert property(@(posedge wb_clk_i) $rose(arst_i) |-> ##1 !wb_inta_o);
assert_a28: assert property(@(posedge wb_clk_i) (wb_rst_i && !scl_pad_i) |=> $fell(wb_rst_i)[->1] |-> !scl_pad_o);
assert_a29: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $fell(scl_pad_i) |-> ##[1:2] wb_inta_o);
assert_a30: assert property(@(posedge wb_clk_i) disable iff (arst_i) ($countones(!scl_pad_i) >= 10) |-> ##[1:2] wb_inta_o);
assert_a31: assert property(@(posedge wb_clk_i) disable iff (arst_i) $rose(!arst_i) |-> ##[1:3] $stable(scl_pad_i));
assert_a32: assert property(@(posedge scl_pad_i) (1, scl_period = $realtime) |-> @(posedge scl_pad_i) (1, scl_period = $realtime - scl_period) |-> scl_period < ($realtime - $last_edge(wb_clk_i))*0.66 |-> ##1 i2c_al);
assert_a33: assert property(@(posedge wb_clk_i) $stable(scl_pad_i) or $changed(scl_pad_i));
assert_a34: assert property(@(posedge wb_clk_i) disable iff (arst_i) (!scl_pad_i && !sda_pad_i) |-> ##1 (scl_pad_i || sda_pad_i));
assert_a35: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $fell(scl_pad_i) |-> ##[0:2] scl_padoen_o);
assert_a36: assert property(@(posedge wb_clk_i) disable iff (arst_i) ($countones(!scl_pad_i) >= 10) |-> ##[1:3] $changed(scl_pad_o));
assert_a37: assert property(@(posedge wb_clk_i) disable iff (arst_i) ($countones(!scl_pad_i) >= 3) |-> ##4 !scl_padoen_o);
assert_a38: assert property(@(posedge wb_clk_i) (wb_cyc_i && wb_stb_i && wb_we_i && $rose(scl_pad_i)) |-> ##[0:1] wb_ack_o);
assert_a39: assert property(@(posedge wb_clk_i) (wb_stb_i && $changed(scl_pad_i)) |-> ##[1:2] wb_ack_o);
assert_a40: assert property(@(posedge wb_clk_i) ($countones(!scl_pad_i) >= 3) |-> ##1 !scl_padoen_o);
assert_a41: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $rose(!wb_rst_i) |-> ##[0:4] ($onehot0(scl_pad_i)));
assert_a42: assert property(@(posedge wb_clk_i) (wb_cyc_i && wb_stb_i && !wb_we_i && $onehot0(scl_pad_i)) |-> ##[0:2] $stable(wb_dat_o));
assert_a43: assert property(@(posedge wb_clk_i) (wb_we_i && $changed(scl_pad_i)) |-> ##[1:3] $changed(wb_dat_o));
assert_a44: assert property(@(posedge wb_clk_i) ($countones(scl_pad_i) >= 8) |-> !wb_inta_o);
assert_a45: assert property(@(posedge wb_clk_i) $fell(scl_pad_i) |-> ##[1:2] wb_inta_o);
assert_a46: assert property(@(posedge arst_i) $stable(scl_pad_i));
assert_a47: assert property(@(posedge wb_clk_i) (wb_we_i && !scl_pad_i) |-> ##1 ($changed(sda_padoen_o)) before (posedge wb_clk_i));
assert_a48: assert property(@(posedge wb_clk_i) (scl_pad_i && $changed(sda_pad_i)) |-> ##[1:$] $stable(scl_padoen_o)[*1]);
assert_a49: assert property(@(posedge wb_clk_i) (wb_cyc_i && $rose(scl_pad_i) && $past(scl_pad_i, 5) == 1'b1) |-> ##1 ($changed(wb_dat_o)));
assert_a50: assert property(@(posedge wb_clk_i) (!arst_i) |-> $stable(scl_pad_i));
assert_a51: assert property(@(posedge wb_clk_i) (scl_pad_i && $changed(wb_adr_i)) |-> $stable(wb_dat_i) until (negedge scl_pad_i));
assert_a52: assert property(@(posedge wb_clk_i) ($fell(wb_rst_i)) |-> ##[1:5] (scl_pad_i == 1'b1));
assert_a53: assert property(@(posedge wb_clk_i) ($fell(arst_i)) |-> ##[1:5] (scl_pad_i == 1'b0 || scl_pad_i == 1'b1));
assert_a54: assert property(@(posedge wb_clk_i) arst_i |-> (scl_pad_o === 1'bz));
assert_a55: assert property(@(posedge wb_clk_i) !scl_padoen_o |=> (scl_pad_o === 1'b0 || scl_pad_o === 1'b1));
assert_a56: assert property(@(posedge wb_clk_i) sto |=> (scl_pad_o === 1'bz));
assert_a57: assert property(@(posedge wb_clk_i) !scl_padoen_o |-> (scl_pad_o === 1'b0 || scl_pad_o === 1'b1));
assert_a58: assert property(@(posedge wb_clk_i) wb_inta_o |-> ##[1:2] (scl_pad_o === 1'b1));
assert_a59: assert property(@(posedge wb_clk_i) wb_rst_i |-> $stable(scl_pad_o));
assert_a60: assert property(@(posedge wb_clk_i) !scl_padoen_o |-> (scl_pad_o === scl_pad_i));
assert_a61: assert property(@(posedge wb_clk_i) disable iff (!rst_i) ($fell(scl_padoen_o) |-> ##1 $stable(scl_pad_o) [*1]) and ($fell(scl_padoen_o) |-> ##2 $stable(scl_pad_o)));
assert_a62: assert property(@(posedge wb_clk_i) (!scl_padoen_o && !sda_padoen_o) [*3] |-> (scl_pad_o === sda_pad_o));
assert_a63: assert property(@(posedge wb_clk_i) (wb_we_i && $changed(wb_dat_i)) |-> ##[1:5] $changed(scl_pad_o));
assert_a64: assert property(@(posedge wb_clk_i) (arst_i == 1'b1) |-> $stable(scl_pad_o));
assert_a65: assert property(@(posedge wb_clk_i) (scl_padoen_o == 1'b1) |-> (scl_pad_o === 1'bz));
assert_a66: assert property(@(posedge wb_clk_i) (arst_i == 1'b1) |-> (scl_pad_o == 1'b1));
assert_a67: assert property(@(posedge wb_clk_i) (wb_cyc_i && wb_stb_i) |-> $stable(scl_pad_o));
assert_a68: assert property(@(posedge wb_clk_i) ((scl_padoen_o == 1'b0) && (sda_padoen_o == 1'b0)) |-> (scl_pad_o == sda_pad_o));
assert_a69: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |-> $stable(scl_pad_o));
assert_a70: assert property(@(posedge wb_clk_i) $changed(scl_padoen_o) |-> $stable(scl_pad_o));
assert_a71: assert property(@(posedge wb_clk_i) (scl_padoen_o == 1'b0) |-> ##[PRERhi:PRERlo] $rose(scl_pad_o));
assert_a72: assert property(@(posedge wb_clk_i) $stable(scl_padoen_o) |-> $stable(scl_pad_o));
assert_a73: assert property(@(posedge wb_clk_i) (!scl_padoen_o && !sda_padoen_o) |-> (scl_pad_o === ~sda_pad_o) [*10]);
assert_a74: assert property(@(posedge wb_clk_i) scl_padoen_o |-> (scl_pad_i === 1'bz));
assert_a75: assert property(@(posedge wb_clk_i) (!scl_padoen_o && wb_we_i) |-> $stable(scl_pad_o));
assert_a76: assert property(@(posedge wb_clk_i) disable iff (scl_padoen_o) !scl_padoen_o |-> $stable(scl_pad_o));
assert_a77: assert property(@(posedge wb_clk_i) disable iff (scl_padoen_o) !scl_padoen_o |-> ##1 $changed(scl_pad_o) |-> $rose(wb_clk_i));
assert_a78: assert property(@(posedge wb_clk_i) scl_padoen_o |=> (scl_pad_o === 1'bz));
assert_a79: assert property(@(posedge wb_clk_i) (CR[3] && !i2c_busy) |-> ##[1:2] (scl_pad_o === 1'b0));
assert_a80: assert property(@(posedge wb_clk_i) scl_padoen_o |-> (scl_pad_o === scl_pad_i));
assert_a81: assert property(@(posedge wb_clk_i) disable iff (arst_i) !arst_i |-> ##[1:5] (scl_pad_o !== 1'bz));
assert_a82: assert property(@(posedge wb_clk_i) $rose(wb_stb_i) |-> (scl_padoen_o == 1'b0)[*8] ##1 (scl_padoen_o != 1'b0));
assert_a83: assert property(@(posedge wb_clk_i) (scl_padoen_o == 1'b1) |=> (scl_pad_o === 1'bz));
assert_a84: assert property(@(posedge wb_clk_i) ((wb_adr_i == 8'h01) && (wb_we_i == 1'b0)) |-> (scl_padoen_o == 1'b1));
assert_a85: assert property(@(posedge wb_clk_i) (wb_ack_o == 1'b1) |-> $stable(scl_padoen_o)[*3]);
assert_a86: assert property(@(posedge wb_clk_i) ((scl_pad_i == 1'b1) && (sda_pad_i == 1'b0)) |-> (scl_padoen_o == 1'b0));
assert_a87: assert property(@(posedge wb_clk_i) $rose(scl_padoen_o) |-> ##[0:9] $fell(scl_padoen_o));
assert_a88: assert property(@(posedge wb_clk_i) (wb_cyc_i == 1'b1) |-> (scl_padoen_o == 1'b0)[*5]);
assert_a89: assert property(@(posedge wb_clk_i) $changed(scl_padoen_o) |-> (wb_cyc_i | wb_stb_i));
assert_a90: assert property(@(posedge wb_clk_i) (scl_padoen_o == 1'b1) |-> (wb_dat_i == 8'h55));
assert_a91: assert property(@(posedge wb_clk_i) (wb_dat_i >= 8'h20 && wb_dat_i <= 8'h7F) |-> scl_padoen_o == 0);
assert_a92: assert property(@(posedge wb_clk_i) scl_padoen_o == 1 |-> scl_pad_o === 1'bz);
assert_a93: assert property(@(posedge wb_clk_i) arst_i |-> scl_padoen_o == 0);
assert_a94: assert property(@(posedge wb_clk_i) $changed(wb_dat_o) |-> $stable(scl_padoen_o));
assert_a95: assert property(@(posedge wb_clk_i) wb_inta_o |-> $stable(scl_padoen_o));
assert_a96: assert property(@(posedge wb_clk_i) $rose(wb_inta_o) |-> ##[1:3] scl_padoen_o == 0);
assert_a97: assert property(@(posedge wb_clk_i) wb_we_i |-> !(scl_padoen_o == 0 && sda_padoen_o == 0));
assert_a98: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |=> scl_padoen_o == 0 [*2]);
assert_a99: assert property(@(posedge wb_clk_i) $rose(arst_i) |=> $stable(scl_padoen_o) throughout (arst_i[->1]));
assert_a100: assert property(@(posedge wb_clk_i) (wb_adr_i >= 8'h00 && wb_adr_i <= 8'h0F) |-> !scl_padoen_o);
assert_a101: assert property(@(posedge wb_clk_i) $rose(scl_padoen_o) |-> wb_ack_o);
assert_a102: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> ##[1:2] !scl_padoen_o);
assert_a103: assert property(@(posedge wb_clk_i) disable iff (!wb_rst_i) !(scl_padoen_o == 0 && sda_padoen_o == 0));
assert_a104: assert property(@(posedge wb_clk_i) $fell(wb_cyc_i) |-> ##[0:1] $rose(scl_padoen_o));
assert_a105: assert property(@(posedge wb_clk_i) $fell(wb_stb_i) |-> ##[0:1] $rose(scl_padoen_o));
assert_a106: assert property(@(posedge wb_clk_i) disable iff (!wb_rst_i) (wb_we_i && scl_padoen_o) |-> !sda_padoen_o);
assert_a107: assert property(@(posedge wb_clk_i) $fell(sda_pad_i) && scl_pad_i |-> ##1 !sda_padoen_o);
assert_a108: assert property(@(posedge scl_pad_i) $changed(sda_pad_i) |-> ##[setup_time:hold_time] $stable(sda_pad_i));
assert_a109: assert property(@(posedge wb_clk_i) $rose(!sda_pad_i) && scl_pad_i |-> ##1 $fell(sda_pad_i) && scl_pad_i);
assert_a110: assert property(@(posedge wb_clk_i) (sda_pad_i && !sda_pad_o && sda_padoen_o) |-> ##1 i2c_al);
assert_a111: assert property(@(posedge wb_clk_i) wb_rst_i |-> ##[0:$] !wb_rst_i |-> ##2 $stable(sda_pad_i));
assert_a112: assert property(@(posedge wb_clk_i) sda_padoen_o |-> ##[0:8] ($stable(sda_pad_i) && sda_pad_i) [*9] ##1 $fell(scl_pad_i));
assert_a113: assert property(@(posedge wb_clk_i) $rose(!arst_i) |-> ##[0:1] sda_pad_i[*2] ##1 $stable(sda_pad_i));
assert_a114: assert property(@(posedge wb_clk_i) $rose(sda_padoen_o) |-> ##[0:2] $rose(sda_pad_i));
assert_a115: assert property(@(posedge wb_clk_i) !sda_pad_i && scl_pad_i |-> i2c_busy);
assert_a116: assert property(@(posedge wb_clk_i) sda_padoen_o |-> $stable(sda_pad_i));
assert_a117: assert property(@(posedge wb_clk_i) !sda_pad_i && !scl_pad_i |-> sda_padoen_o throughout scl_pad_i[->1]);
assert_a118: assert property(@(posedge wb_clk_i) $rose(scl_pad_i) |-> ##1 $stable(sda_pad_i));
assert_a119: assert property(@(posedge scl_pad_i) $fell(sda_pad_i) |-> ##[0:7] $stable(sda_pad_i)[*8]);
assert_a120: assert property(@(posedge wb_clk_i) ($changed(sda_pad_i) && scl_pad_i == 1 && wb_we_i == 1) |-> ##[1:3] wb_inta_o == 1);
assert_a121: assert property(@(posedge scl_pad_i) $fell(sda_pad_i) |-> ##[0:7] $stable(sda_pad_i)[*8] ##1 sda_pad_i == 1);
assert_a122: assert property(@(posedge wb_clk_i) ($changed(sda_pad_i)) |-> ##[setup_time:hold_time] $stable(scl_pad_i));
assert_a123: assert property(@(posedge wb_clk_i) $rose(arst_i) |-> $stable(sda_pad_i) throughout (arst_i));
assert_a124: assert property(@(posedge wb_clk_i) (sda_padoen_o == 0) |-> sda_pad_i == sda_pad_o);
assert_a125: assert property(@(posedge wb_clk_i) $fell(sda_pad_i) |-> ##[0:7] $stable(sda_pad_i)[*8] |-> ##1 i2c_al == 1);
assert_a126: assert property(@(posedge wb_clk_i) (wb_we_i == 0 && $changed(sda_pad_i)) |-> $stable(wb_dat_o) until $rose(scl_pad_i));
assert_a127: assert property(@(posedge wb_clk_i) (!arst_i) |-> $stable(sda_pad_i) within ([*1:$]));
assert_a128: assert property(@(posedge wb_clk_i) (sda_pad_i == 0 && sda_padoen_o == 1) |-> ##[1:3] i2c_al == 1);
assert_a129: assert property(@(posedge scl_pad_i) sda_pad_i == 0 |-> ##1 rxr[0] == 0);
assert_a130: assert property(@(posedge wb_clk_i) ($changed(sda_pad_i)) |-> $fell(scl_pad_i) within ([*1:$]));
assert_a131: assert property(@(posedge wb_clk_i) $rose(!arst_i) |-> $stable(sda_pad_i)[*1] ##0 $stable(sda_padoen_o));
assert_a132: assert property(@(posedge wb_clk_i) $changed(sda_pad_i) && !scl_pad_i |-> ##[0:1] !$stable(sda_pad_i));
assert_a133: assert property(@(posedge wb_clk_i) $fell(sda_pad_i) && scl_pad_i |-> ##[0:49] sda_pad_i || ##50 $rose(sda_pad_i));
assert_a134: assert property(@(posedge wb_clk_i) !sda_pad_i && !scl_pad_i |-> scl_padoen_o == 1'b1);
assert_a135: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> ##[0:2] !$isunknown(sda_pad_i) && sda_pad_i inside {0,1});
assert_a136: assert property(@(posedge wb_clk_i) arst_i |-> $stable(sda_pad_i));
assert_a137: assert property(@(posedge wb_clk_i) $rose(sda_pad_i) && scl_pad_i |-> ##[1:$] $past(scl_pad_i) && !scl_pad_i);
assert_a138: assert property(@(posedge wb_clk_i) !sda_pad_i && !sda_padoen_o |-> $isunknown(sda_pad_i) == 0 && sda_pad_i inside {0,1});
assert_a139: assert property(@(posedge scl_pad_i) sda_pad_i |-> $past(sda_pad_i) == 1'b1);
assert_a140: assert property(@(posedge wb_clk_i) sda_padoen_o |-> $isunknown(sda_pad_i) == 1);
assert_a141: assert property(@(posedge wb_clk_i) sda_pad_i && scl_pad_i [*4] |-> ##1 sda_pad_o == 1'b1);
assert_a142: assert property(@(posedge wb_clk_i) disable iff (arst_i) (!sda_padoen_o) |-> !$isunknown(sda_pad_o));
assert_a143: assert property(@(posedge wb_clk_i) disable iff (arst_i) (sda_padoen_o) |-> ##1 $isunknown(sda_pad_o));
assert_a144: assert property(@(posedge wb_clk_i) disable iff (arst_i) (wb_ack_o) |-> $stable(sda_pad_o));
assert_a145: assert property(@(posedge wb_clk_i) disable iff (arst_i) (!sda_padoen_o) |-> (sda_pad_o && $fell(scl_pad_i)) ##1 (!sda_pad_o && scl_pad_i) ##1 (sda_pad_o[0] == $past(sda_pad_o,1))[*8]);
assert_a146: assert property(@(posedge wb_clk_i) disable iff (arst_i) (!wb_we_i && sda_padoen_o) |-> $isunknown(sda_pad_o));
assert_a147: assert property(@(posedge wb_clk_i) disable iff (arst_i) (wb_we_i && $changed(wb_dat_i)) |=> (sda_pad_o == ~sda_pad_i));
assert_a148: assert property(@(posedge wb_clk_i) ($rose(!arst_i)) |-> ##[0:2] !$isunknown(sda_pad_o));
assert_a149: assert property(@(posedge wb_clk_i) disable iff (arst_i) (!scl_pad_i) |-> $stable(sda_pad_o) until (scl_pad_i));
assert_a150: assert property(@(posedge wb_clk_i) ($rose(!arst_i)) |-> ##[1:2] !$isunknown(sda_pad_o));
assert_a151: assert property(@(posedge wb_clk_i) disable iff (arst_i) (!sda_padoen_o) |-> (sda_pad_o === 1'b0 || sda_pad_o === 1'b1));
assert_a152: assert property(@(posedge wb_clk_i) disable iff (arst_i) (wb_we_i |-> ##[1:3] $isunknown(sda_pad_o) == 0));
assert_a153: assert property(@(posedge wb_clk_i) disable iff (arst_i) (!sda_padoen_o && (sda_pad_i !== sda_pad_o) |-> sda_pad_o == $past(sda_pad_o)));
assert_a154: assert property(@(posedge wb_clk_i) disable iff (arst_i) (wb_cyc_i && wb_stb_i |-> $stable(sda_pad_o) until wb_ack_o));
assert_a155: assert property(@(posedge wb_clk_i) (arst_i |-> $stable(sda_pad_o)));
assert_a156: assert property(@(posedge wb_clk_i) disable iff (arst_i) (wb_ack_o |-> $stable(sda_pad_o) throughout wb_ack_o[->1]));
assert_a157: assert property(@(posedge wb_clk_i) disable iff (arst_i) (wb_cyc_i && wb_stb_i && !sda_padoen_o |-> ##1 $stable(sda_pad_o)[*2]));
assert_a158: assert property(@(posedge wb_clk_i) disable iff (arst_i) ((sda_pad_i !== sda_pad_o) |-> ##[1:2] !sda_padoen_o));
assert_a159: assert property(@(posedge wb_clk_i) disable iff (arst_i) ($changed(sda_padoen_o) |-> ##1 $stable(sda_pad_o)[*1]));
assert_a160: assert property(@(posedge wb_clk_i) disable iff (arst_i) (sda_padoen_o |-> sda_pad_o === 1'bz));
assert_a161: assert property(@(posedge wb_clk_i) disable iff (arst_i) ($fell(sda_padoen_o) |-> ##1 $stable(sda_pad_o)[*2]));
assert_a162: assert property(@(posedge wb_clk_i) disable iff (sda_padoen_o) (wb_we_i) |=> (sda_pad_o == $past(wb_dat_i)));
assert_a163: assert property(@(posedge wb_clk_i) disable iff (sda_padoen_o) (!wb_we_i) |-> (sda_pad_o == wb_dat_i));
assert_a164: assert property(@(posedge wb_clk_i) (arst_i) |-> $stable(sda_pad_o));
assert_a165: assert property(@(posedge wb_clk_i) disable iff (sda_padoen_o) $changed(sda_pad_i) |-> $stable(sda_pad_o));
assert_a166: assert property(@(posedge wb_clk_i) disable iff (wb_ack_o || sda_padoen_o) (wb_we_i && !wb_ack_o) |-> $stable(sda_pad_o));
assert_a167: assert property(@(posedge wb_clk_i) (sda_padoen_o) |-> (sda_pad_o === 1'bz));
assert_a168: assert property(@(posedge wb_clk_i) (~sda_padoen_o) |-> (sda_pad_o inside {0, 1}));
assert_a169: assert property(@(posedge wb_clk_i) disable iff (sda_padoen_o) ($rose(~arst_i)) |=> (sda_pad_o inside {0, 1}));
assert_a170: assert property(@(posedge wb_clk_i) (wb_adr_i == `I2C_ADDR) |-> ##1 (sda_padoen_o == 1'b0));
assert_a171: assert property(@(posedge wb_clk_i) wb_ack_o |-> $stable(sda_padoen_o) throughout (sda_pad_i[->1]));
assert_a172: assert property(@(posedge wb_clk_i) (!wb_cyc_i && !wb_stb_i) |-> ##[1:2] (sda_padoen_o == 1'b1));
assert_a173: assert property(@(posedge wb_clk_i) (scl_padoen_o == 1'b0) |-> (sda_padoen_o != 1'b1));
assert_a174: assert property(@(posedge wb_clk_i) (wb_we_i && (wb_adr_i inside {[`I2C_CTRL_LO:`I2C_CTRL_HI]})) |-> ##[1:2] (sda_padoen_o == 1'b1));
assert_a175: assert property(@(posedge wb_clk_i) wb_we_i |-> ##[1:2] $stable(sda_padoen_o));
assert_a176: assert property(@(posedge wb_clk_i) $rose(wb_dat_o) |-> $stable(sda_padoen_o));
assert_a177: assert property(@(posedge wb_clk_i) (scl_padoen_o == 1'b1)[*3] |-> ##1 (sda_padoen_o == 1'b1));
assert_a178: assert property(@(posedge wb_clk_i) arst_i |-> $stable(sda_padoen_o));
assert_a179: assert property(@(posedge wb_clk_i) $rose(wb_rst_i) |-> (sda_padoen_o == 1'b1) until (wb_stb_i && wb_cyc_i));
assert_a180: assert property(@(posedge wb_clk_i) (sda_padoen_o == 1'b1) |-> sda_pad_o == 1'b0);
assert_a181: assert property(@(posedge wb_clk_i) (wb_inta_o == 1'b1) |-> sda_padoen_o == 1'b1);
assert_a182: assert property(@(posedge wb_clk_i) (sda_pad_i != sda_pad_o) |=> sda_padoen_o == 1'b0);
assert_a183: assert property(@(posedge wb_clk_i) $rose(wb_inta_o) |-> sda_padoen_o == 1'b1);
assert_a184: assert property(@(posedge wb_clk_i) $fell(rst_i) |-> sda_padoen_o == 1'b1 until $rose(rst_i));
assert_a185: assert property(@(posedge wb_clk_i) $rose(wb_dat_i) |-> sda_padoen_o == !sda_pad_o);
assert_a186: assert property(@(posedge wb_clk_i) ($rose(scl_pad_o) && $rose(sda_pad_o)) |-> sda_padoen_o == 1'b0 throughout [0:1]);
assert_a187: assert property(@(posedge wb_clk_i) (scl_pad_o == 1'b0 && wb_stb_i == 1'b1) |-> sda_padoen_o == 1'b0);
assert_a188: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> sda_padoen_o == 1'b1 until $rose(arst_i));
assert_a189: assert property(@(posedge wb_clk_i) (i2c_busy == 1'b0) |-> sda_padoen_o == !sda_pad_i);
assert_a190: assert property(@(posedge wb_clk_i) wb_inta_o |-> (sda_padoen_o == expected_sda_oen));
assert_a191: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |-> ##[1:2] $fell(sda_padoen_o));
assert_a192: assert property(@(posedge wb_clk_i) (scl_pad_o && wb_ack_o) |-> (master_transmitting ? sda_padoen_o == 1'b0 : sda_padoen_o == 1'b1));
assert_a193: assert property(@(posedge wb_clk_i) $rose(arst_i) |-> sda_padoen_o == 1'b1);
assert_a194: assert property(@(posedge wb_clk_i) (i2c_busy && $stable(scl_pad_i)) |-> $stable(sda_padoen_o));
assert_a195: assert property(@(posedge wb_clk_i) (wb_we_i && $past(wb_we_i)) |-> $stable(sda_padoen_o));
assert_a196: assert property(@(posedge wb_clk_i) (wb_dat_i == START_CMD_PATTERN) |-> ##[1:$] (sda_padoen_o == 1'b0) before ($fell(scl_pad_o)));
assert_a197: assert property(@(posedge wb_clk_i) $rose(wb_ack_o) |-> ##1 (sda_padoen_o == 1'b0));
assert_a198: assert property(@(posedge wb_clk_i) $fell(wb_stb_i) |-> (sda_padoen_o == 1'b0) [*3]);
assert_a199: assert property(@(posedge wb_clk_i) ($rose(wb_cyc_i && wb_stb_i) |-> sda_padoen_o !== $past(sda_padoen_o)) |-> scl_padoen_o);
assert_a200: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($rose(wb_ack_o) |=> !wb_ack_o));
assert_a201: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $rose(wb_ack_o) |-> (wb_cyc_i && wb_stb_i && $stable(wb_clk_i)));
assert_a202: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_ack_o && (!$stable(wb_cyc_i) || !$stable(wb_stb_i))) |=> !wb_ack_o);
assert_a203: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_ack_o && $past(wb_ack_o)) |-> !(wb_cyc_i && wb_stb_i));
assert_a204: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && wb_stb_i && !$past(wb_ack_o)) |=> wb_ack_o);
assert_a205: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($fell(wb_cyc_i) || $fell(wb_stb_i)) |-> !wb_ack_o);
assert_a206: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $rose(wb_ack_o) |-> (wb_cyc_i && wb_stb_i));
assert_a207: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($changed(arst_i) && !wb_rst_i) |-> ##1 $stable(wb_ack_o) or $rose(wb_ack_o) == (wb_cyc_i && wb_stb_i && !$past(wb_ack_o)));
assert_a208: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_cyc_i || !wb_stb_i) |-> !wb_ack_o);
assert_a209: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($rose(wb_ack_o) |-> (wb_cyc_i && wb_stb_i && !$past(wb_ack_o))));
assert_a210: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_cyc_i && wb_stb_i && !$past(wb_ack_o)) |-> wb_ack_o);
assert_a211: assert property(@(posedge wb_clk_i) disable iff (!wb_rst_i) arst_i |-> !wb_ack_o);
assert_a212: assert property(@(posedge wb_clk_i) (wb_rst_i || arst_i) |-> $stable(wb_ack_o));
assert_a213: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_ack_o && wb_cyc_i && wb_stb_i) |=> !wb_ack_o);
assert_a214: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) $stable(wb_clk_i) |-> $stable(wb_ack_o));
assert_a215: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (!wb_cyc_i || !wb_stb_i) |-> !wb_ack_o);
assert_a216: assert property(@(posedge wb_clk_i) wb_rst_i |-> !wb_ack_o);
assert_a217: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) $changed(wb_adr_i) or $changed(wb_dat_i) or $changed(wb_we_i) |-> !(wb_cyc_i && wb_stb_i) || $stable(wb_ack_o));
assert_a218: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) !(wb_cyc_i && wb_stb_i) |-> !wb_ack_o);
assert_a219: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) wb_cyc_i && !wb_stb_i |-> !wb_ack_o);
assert_a220: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!$rose(wb_cyc_i && wb_stb_i)) |=> $stable(wb_ack_o));
assert_a221: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) wb_ack_o |=> !wb_ack_o || (wb_cyc_i && wb_stb_i));
assert_a222: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && wb_stb_i && !wb_ack_o) |=> wb_ack_o);
assert_a223: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) wb_ack_o && !(wb_cyc_i && wb_stb_i) |=> !wb_ack_o until (wb_cyc_i && wb_stb_i));
assert_a224: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $changed(scl_pad_i) or $changed(sda_pad_i) or $changed(wb_inta_o) |-> $stable(wb_ack_o));
assert_a225: assert property(@(posedge wb_clk_i) $fell(arst_i) |-> ##[1:3] (wb_stb_i && wb_cyc_i && !wb_we_i && (wb_adr_i inside {3'b000, 3'b001, 3'b010, 3'b011}) |-> ##2 wb_ack_o));
assert_a226: assert property(@(posedge wb_clk_i) $rose(arst_i) |=> (wb_dat_o == 8'h00) until_with ($fell(arst_i) ##1 1));
assert_a227: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && (wb_adr_i == 3'b000)) |-> ##2 (wb_dat_o == status_reg));
assert_a228: assert property(@(posedge wb_clk_i) (arst_i) |-> (wb_dat_o == 8'h00) until (!arst_i));
assert_a229: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_cyc_i && wb_we_i && (wb_adr_i == 3'b011)) |-> ##2 (int_mask_reg == $past(wb_dat_i, 2)));
assert_a230: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_cyc_i && !wb_we_i) |-> ##2 ( (wb_adr_i == 3'b000) -> (wb_dat_o == status_reg) and (wb_adr_i == 3'b001) -> (wb_dat_o == data_reg) and (wb_adr_i == 3'b010) -> (wb_dat_o == control_reg) and (wb_adr_i == 3'b011) -> (wb_dat_o == int_mask_reg) ));
assert_a231: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_cyc_i && !wb_we_i) |-> ##2 ( (wb_adr_i[2:0] == 3'b000) -> (wb_dat_o == $past(wb_dat_o, 2) when extended_addr_bits !== 0) ));
assert_a232: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_cyc_i && (wb_adr_i == 3'b111)) |-> (!wb_ack_o throughout ##[0:2]) and (wb_dat_o == 8'h00));
assert_a233: assert property(@(posedge wb_clk_i) (wb_rst_i) |=> (wb_dat_o == $past(wb_dat_o)));
assert_a234: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_cyc_i && !wb_we_i && (wb_adr_i == 3'b000)) |-> ##2 (wb_dat_o == $past(status_reg, 2)));
assert_a235: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i inside {3'b000, 3'b001, 3'b010, 3'b011} && wb_stb_i && wb_cyc_i) ##1 (wb_adr_i inside {3'b000, 3'b001, 3'b010, 3'b011} && $changed(wb_adr_i) && wb_stb_i && wb_cyc_i) |-> ##[1:2] (wb_dat_o == register_array[wb_adr_i]));
assert_a236: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i == 3'b111 && !wb_we_i && wb_cyc_i) |-> ##[1:3] (wb_dat_o == read_only_reg));
assert_a237: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i == 3'b001 && wb_we_i && wb_stb_i && wb_cyc_i) |-> ##[1:2] ($past(wb_dat_i, 1) == ctr_reg_value));
assert_a238: assert property(@(posedge wb_clk_i) (wb_rst_i) |-> (wb_dat_o == 8'h00) [*1:$]);
assert_a239: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i == 3'b100 && wb_we_i && wb_stb_i) |-> ##1 (internal_reg_4 == $past(wb_dat_i, 1)));
assert_a240: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && $changed(wb_adr_i)) |-> !wb_ack_o until ($stable(wb_adr_i) [*1]));
assert_a241: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i == 3'b100 && !wb_we_i && wb_stb_i && wb_cyc_i) |-> ##[1:2] (wb_dat_o == int_status_reg));
assert_a242: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i inside {3'b100, 3'b101, 3'b110, 3'b111} && wb_we_i && wb_stb_i && wb_cyc_i) |-> !wb_ack_o [*1:$]);
assert_a243: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i inside {3'b000, 3'b001, 3'b010, 3'b011} && wb_stb_i && wb_cyc_i) |-> ##[1:2] (wb_dat_o == register_array[wb_adr_i]));
assert_a244: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i == 3'b010 && !wb_we_i && wb_stb_i && wb_cyc_i) |-> ##[1:2] $stable(wb_dat_o));
assert_a245: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && wb_stb_i && wb_adr_i inside {3'b000, 3'b001, 3'b010, 3'b011} ##1 wb_cyc_i && wb_stb_i && wb_adr_i inside {3'b000, 3'b001, 3'b010, 3'b011} && $changed(wb_adr_i)) |-> ##[1:2] $changed(wb_dat_o));
assert_a246: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i == 3'b110 && !wb_stb_i) |-> $stable(wb_dat_o));
assert_a247: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($changed(wb_adr_i) && !wb_stb_i ##1 wb_stb_i) |-> ##[1:2] $changed(wb_dat_o));
assert_a248: assert property(@(posedge wb_clk_i) (wb_rst_i && $changed(wb_adr_i)) |-> $stable(wb_dat_o) throughout (!wb_rst_i)[->1]);
assert_a249: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_adr_i == 3'b000 && wb_cyc_i && wb_stb_i ##1 wb_adr_i == 3'b001 && wb_cyc_i && wb_stb_i) |-> ##[1:2] $changed(wb_dat_o));
assert_a250: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($changed(wb_adr_i) && !wb_stb_i) |-> ($stable(wb_dat_o) && $stable(wb_ack_o)));
assert_a251: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && wb_stb_i && !wb_we_i && $stable(wb_adr_i)[*2]) |-> ($stable(wb_dat_o) && ##1 wb_ack_o));
assert_a252: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($changed(wb_adr_i) && !wb_cyc_i ##1 wb_cyc_i) |-> ##[1:2] wb_ack_o);
assert_a253: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |-> ##[1:$] $stable(wb_rst_i));
assert_a254: assert property(@(posedge wb_clk_i) $rose(wb_ack_o) |-> ##1 $stable(wb_ack_o) [*1:$] until !wb_cyc_i or !wb_stb_i);
assert_a255: assert property(@(posedge wb_clk_i) (wb_we_i && wb_stb_i) |-> ##1 (wb_we_i && wb_stb_i));
assert_a256: assert property(@(posedge wb_clk_i) $fell(wb_cyc_i || wb_stb_i) |=> !wb_ack_o);
assert_a257: assert property(@(posedge wb_clk_i or posedge wb_rst_i) if (wb_rst_i == ARST_LVL) (!wb_ack_o && (wb_dat_o == 0)));
assert_a258: assert property(@(posedge wb_clk_i) $changed(sda_pad_o) |-> $past(!$stable(sda_pad_o)));
assert_a259: assert property(@(posedge wb_clk_i) (wb_ack_o && $past(wb_ack_o, 1)) |-> $past(!wb_stb_i, 1));
assert_a260: assert property(@(posedge wb_clk_i) $changed(sda_pad_o) |-> $past($stable(sda_pad_o)));
assert_a261: assert property(@(posedge wb_clk_i) (wb_rst_i == ARST_LVL) |-> !wb_ack_o);
assert_a262: assert property(@(posedge wb_clk_i) (!wb_we_i && wb_stb_i && !wb_ack_o) |-> ##[0:5] $stable(wb_dat_o) until wb_ack_o);
assert_a263: assert property(@(posedge wb_clk_i or posedge wb_rst_i) if (wb_rst_i == ARST_LVL) !wb_ack_o);
assert_a264: assert property(@(posedge wb_clk_i) (wb_cyc_i && wb_stb_i && !wb_ack_o) |=> wb_ack_o);
assert_a265: assert property(@(posedge wb_clk_i) $stable(wb_adr_i) |-> ##1 $stable(wb_dat_o));
assert_a266: assert property(@(posedge wb_clk_i) (wb_cyc_i && wb_stb_i) |-> ##1 wb_ack_o && ##1 !wb_ack_o);
assert_a267: assert property(@(posedge wb_clk_i or posedge arst_i) if (arst_i) (wb_ack_o == 0 && wb_dat_o == 0 && wb_inta_o == 0));
assert_a268: assert property(@(posedge wb_clk_i) (wb_ack_o && (!wb_cyc_i || !wb_stb_i)) |-> ##1 !wb_ack_o);
assert_a269: assert property(@(posedge wb_clk_i) $rose(interrupt_condition) |-> ##[1:2] wb_inta_o);
assert_a270: assert property(@(posedge wb_clk_i) wb_we_i |-> ##1 i2c_control_data == $past(wb_dat_i));
assert_a271: assert property(@(posedge wb_clk_i) disable iff (!wb_cyc_i || !wb_stb_i) !wb_ack_o |=> (wb_cyc_i && wb_stb_i) |-> ##1 wb_ack_o);
assert_a272: assert property(@(posedge wb_clk_i) $rose(interrupt_condition) |-> ##1 wb_inta_o);
assert_a273: assert property(@(posedge wb_clk_i) $changed(i2c_start_stop_condition) |-> ##1 $stable(scl_padoen_o) && $stable(sda_padoen_o));
assert_a274: assert property(@(posedge wb_clk_i) $stable(scl_pad_i) && $stable(sda_pad_i) |-> ##1 $stable(scl_pad_i) && $stable(sda_pad_i));
assert_a275: assert property(@(posedge wb_clk_i) (wb_adr_i == 3'b000) |-> ##1 wb_dat_o == prer[7:0]);
assert_a276: assert property(@(posedge wb_clk_i) i2c_active |-> ##1 $stable(scl_pad_o) && $stable(sda_pad_o));
assert_a277: assert property(@(posedge wb_clk_i) !(wb_cyc_i && wb_stb_i) |-> !wb_ack_o);
assert_a278: assert property(@(posedge wb_clk_i) $stable(wb_adr_i) |-> $stable(wb_dat_o));
assert_a279: assert property(@(posedge wb_clk_i) !(wb_adr_i inside {3'b000, 3'b001, 3'b010, 3'b011, 3'b100, 3'b101, 3'b110, 3'b111}) |-> $stable(wb_dat_o));
assert_a280: assert property(@(posedge wb_clk_i) $changed(addressing_mode) |-> $rose(wb_cyc_i));
assert_a281: assert property(@(posedge wb_clk_i) wb_ack_o |-> $past(wb_cyc_i && wb_stb_i, 1));
assert_a282: assert property(@(posedge wb_clk_i) (!scl_pad_i || !sda_pad_i) |-> ##[1:2] wb_inta_o);
assert_a283: assert property(@(posedge wb_clk_i) interrupt_condition |-> ##[1:2] wb_inta_o);
assert_a284: assert property(@(posedge wb_clk_i) $changed(scl_padoen_o) || $changed(sda_padoen_o) |-> $rose(wb_cyc_i & wb_stb_i));
assert_a285: assert property(@(posedge wb_clk_i) $rose(interrupt_cleared) |-> ##1 wb_inta_o && ##1 !wb_inta_o);
assert_a286: assert property(@(posedge wb_clk_i) (wb_adr_i == 3'b000) |-> ##1 (wb_dat_o == prer[7:0]));
assert_a287: assert property(@(posedge wb_clk_i) $stable(wb_adr_i) |-> $stable(wb_dat_o) until $changed(wb_adr_i));
assert_a288: assert property(@(posedge wb_clk_i) $fell(wb_stb_i) && $past(wb_ack_o) |-> !wb_ack_o);
assert_a289: assert property(@(posedge wb_clk_i) $changed(wb_adr_i) |-> ##1 $stable(wb_dat_o) until $changed(wb_adr_i));
assert_a290: assert property(@(posedge wb_clk_i) (wb_rst_i == 1'b1) |-> (wb_cyc_i == 1'b0));
assert_a291: assert property(@(posedge wb_clk_i) (wb_rst_i == 1'b1) |-> not (wb_cyc_i == 1'b1));
assert_a292: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i == 1'b1) |-> ##[1:2] (wb_ack_o == 1'b1));
assert_a293: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $rose(wb_cyc_i) |-> (wb_cyc_i == 1'b1) throughout (wb_ack_o == 1'b1)[->1]);
assert_a294: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $fell(wb_cyc_i) |-> $stable(wb_dat_o) until $rose(wb_cyc_i));
assert_a295: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_stb_i == 1'b1) |-> (wb_cyc_i == 1'b1));
assert_a296: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i == 1'b1 && wb_stb_i == 1'b0) |-> (wb_ack_o == 1'b0));
assert_a297: assert property(@(posedge wb_clk_i) (wb_rst_i == 1'b1) |=> (wb_cyc_i == 1'b0));
assert_a298: assert property(@(posedge wb_clk_i or posedge arst_i) (arst_i == 1'b1) |-> not (wb_cyc_i == 1'b1));
assert_a299: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i == 1'b1 && wb_stb_i == 1'b1) |-> ##[1:3] (wb_ack_o == 1'b1));
assert_a300: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) wb_cyc_i |-> ##1 wb_cyc_i);
assert_a301: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && !wb_ack_o) |=> wb_cyc_i until_with wb_ack_o);
assert_a302: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) wb_cyc_i |-> (wb_we_i || wb_stb_i));
assert_a303: assert property(@(posedge wb_clk_i) $rose(!arst_i) |=> $stable(wb_cyc_i));
assert_a304: assert property(@(posedge wb_clk_i) wb_rst_i |-> !wb_cyc_i);
assert_a305: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) wb_cyc_i |-> wb_stb_i);
assert_a306: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $rose(wb_cyc_i) |-> $past(!wb_cyc_i));
assert_a307: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_cyc_i && !$rose(wb_inta_o)) |-> !wb_inta_o);
assert_a308: assert property(@(posedge wb_clk_i) (wb_cyc_i) |-> !$isunknown(wb_adr_i));
assert_a309: assert property(@(posedge wb_clk_i) (arst_i) |-> !wb_cyc_i);
assert_a310: assert property(@(posedge wb_clk_i) disable iff (arst_i) $stable(wb_cyc_i));
assert_a311: assert property(@(posedge wb_clk_i) (wb_ack_o) |-> ##1 !wb_cyc_i);
assert_a312: assert property(@(posedge wb_clk_i) (wb_ack_o) |=> !wb_cyc_i);
assert_a313: assert property(@(posedge wb_clk_i) (wb_cyc_i) |-> ##[1:2] wb_ack_o);
assert_a314: assert property(@(posedge wb_clk_i) (wb_cyc_i) |-> wb_stb_i);
assert_a315: assert property(@(posedge wb_clk_i) (!wb_stb_i) |-> !wb_cyc_i);
assert_a316: assert property(@(posedge wb_clk_i) (wb_cyc_i && wb_we_i) |-> ##[1:2] ($past(wb_dat_i,1) == $past(wb_dat_i,2)));
assert_a317: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_we_i && wb_stb_i && wb_ack_o) |-> ##[1:2] $stable(wb_dat_i));
assert_a318: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_dat_i == 8'hFF && wb_we_i) |-> ##1 (wb_dat_o == 8'hFF));
assert_a319: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!$stable(wb_dat_i) && !wb_stb_i) |-> !wb_ack_o);
assert_a320: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) 1'b1 |-> (wb_dat_i == wb_dat_i[7:0]));
assert_a321: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($fell(wb_rst_i)) |-> ##3 (wb_dat_i >= 8'h20 && wb_dat_i <= 8'hDF));
assert_a322: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_we_i && wb_stb_i) |-> ##[1:2] $stable(wb_dat_i));
assert_a323: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($rose(wb_stb_i) && !$past($stable(wb_dat_i), 1)) |-> !wb_ack_o);
assert_a324: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_rst_i) |-> (wb_dat_i >= 8'h00 && wb_dat_i <= 8'hFF));
assert_a325: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && wb_stb_i) |-> $stable(wb_dat_i) [*1]);
assert_a326: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_dat_i == 8'hFF && wb_we_i && wb_stb_i) |-> ##2 (prer[15:8] == 8'hFF));
assert_a327: assert property(@(posedge wb_clk_i) (!wb_ack_o && wb_stb_i) |-> $stable(wb_dat_i) until (wb_ack_o));
assert_a328: assert property(@(posedge wb_clk_i) (wb_stb_i && wb_adr_i == 3'b101) |-> ##1 (prer[15:8] == $past(wb_dat_i)));
assert_a329: assert property(@(posedge wb_clk_i) (wb_dat_i inside {[8'hA0:8'hAF]}) |=> (wb_dat_i inside {[8'h80:8'hAF]}));
assert_a330: assert property(@(posedge wb_clk_i) (!wb_stb_i) |-> $stable(wb_dat_i));
assert_a331: assert property(@(posedge wb_clk_i) (wb_rst_i) |-> $stable(wb_dat_o));
assert_a332: assert property(@(posedge wb_clk_i) (wb_dat_i == 8'h00 && wb_we_i && wb_stb_i) |-> ##[0:1] !wb_ack_o until (wb_ack_o));
assert_a333: assert property(@(posedge wb_clk_i) (arst_i) |-> !(wb_cyc_i && wb_stb_i) throughout (##[1:$] (wb_cyc_i && wb_stb_i)[->1]));
assert_a334: assert property(@(posedge wb_clk_i) (wb_dat_i inside {[8'h00:8'h7F]}) |=> (wb_dat_i inside {[8'h00:8'hFF]}));
assert_a335: assert property(@(posedge wb_clk_i) (wb_we_i && wb_stb_i) |-> ##1 $stable(wb_dat_i)[*2]);
assert_a336: assert property(@(posedge wb_clk_i) (wb_cyc_i && !wb_stb_i) |-> $stable(wb_dat_i) until (wb_stb_i));
assert_a337: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_we_i && wb_stb_i && wb_cyc_i) |-> $stable(wb_dat_i) throughout (wb_ack_o [->1]));
assert_a338: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) ((wb_dat_i >= 8'h00) && (wb_dat_i <= 8'h7F)) |=> (wb_dat_i >= 8'h00) && (wb_dat_i <= 8'hFF));
assert_a339: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_cyc_i && wb_stb_i && wb_we_i) |-> $stable(wb_dat_i) throughout (wb_ack_o [->1]));
assert_a340: assert property(@(posedge wb_clk_i) (arst_i) |-> !wb_ack_o);
assert_a341: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_cyc_i && wb_stb_i) |-> ##1 $past(wb_dat_i) == wb_dat_i);
assert_a342: assert property(@(posedge wb_clk_i) ($fell(wb_rst_i)) |-> ##2 (wb_dat_i >= 8'h10) && (wb_dat_i <= 8'hEF));
assert_a343: assert property(@(posedge wb_clk_i) (wb_rst_i) |-> $stable(wb_dat_i));
assert_a344: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_adr_i == 3'b101 && wb_we_i && wb_cyc_i && wb_stb_i) |-> ##[1:2] wb_ack_o);
assert_a345: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_cyc_i) |=> $stable(wb_dat_o));
assert_a346: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && $changed(sda_pad_i)) |-> $stable(wb_dat_o));
assert_a347: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) wb_we_i |-> (wb_dat_o === 8'hzz));
assert_a348: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && !wb_ack_o) |-> $stable(wb_dat_o));
assert_a349: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_inta_o && !wb_we_i) ##1 wb_ack_o |-> (wb_dat_o == 8'hXX));
assert_a350: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && $fell(wb_stb_i)) |=> $stable(wb_dat_o));
assert_a351: assert property(@(posedge wb_clk_i) (wb_rst_i && !wb_we_i && wb_stb_i) |-> (wb_dat_o == 8'h00));
assert_a352: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && $changed(wb_adr_i)) ##1 wb_ack_o |=> ##2 $changed(wb_dat_o));
assert_a353: assert property(@(posedge wb_clk_i) $fell(arst_i) && !wb_cyc_i |=> ##2 (wb_dat_o == 8'hFF));
assert_a354: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i) ##1 wb_ack_o |-> (wb_dat_o >= 8'h00 && wb_dat_o <= 8'hFF));
assert_a355: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && $changed(wb_adr_i)) |=> ##3 $changed(wb_dat_o));
assert_a356: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_inta_o) ##1 (!wb_we_i) |-> ##[1:2] (wb_dat_o == 8'hAA));
assert_a357: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($rose(!wb_we_i)) |-> $stable(wb_dat_o) until (wb_stb_i && wb_cyc_i && !wb_we_i));
assert_a358: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && wb_cyc_i) ##0 wb_ack_o |=> $changed(wb_dat_o));
assert_a359: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && wb_cyc_i) |-> $stable(wb_dat_o) or (wb_ack_o && $changed(wb_dat_o)));
assert_a360: assert property(@(posedge wb_clk_i) (wb_rst_i) |-> (wb_dat_o == 8'h00) until (!wb_rst_i && wb_cyc_i && wb_ack_o));
assert_a361: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_inta_o) ##1 (!wb_we_i && wb_stb_i) |-> (wb_dat_o == 8'hAA));
assert_a362: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && wb_cyc_i && !wb_ack_o) |=> $stable(wb_dat_o) throughout (wb_ack_o [->1]));
assert_a363: assert property(@(posedge wb_clk_i) $rose(wb_rst_i) |=> (wb_dat_o == 8'h00));
assert_a364: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && wb_ack_o) |-> (wb_dat_o inside {[8'h00:8'hFF]}));
assert_a365: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_we_i && wb_stb_i && wb_cyc_i && wb_ack_o) ##1 (!wb_we_i && wb_stb_i && wb_cyc_i && wb_ack_o) |-> (wb_dat_o == $past(wb_dat_i)));
assert_a366: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && wb_cyc_i)[*2] |-> $stable(wb_dat_o) throughout (##[1:1] (!wb_we_i && wb_stb_i && wb_cyc_i)));
assert_a367: assert property(@(posedge wb_clk_i) $rose(wb_rst_i) |=> ##1 (wb_dat_o == 8'h00));
assert_a368: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_ack_o && wb_stb_i) |-> ##1 $stable(wb_dat_o));
assert_a369: assert property(@(posedge wb_clk_i or posedge arst_i) if (arst_i) (wb_dat_o == 8'h00) within [0:1] clk);
assert_a370: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $fell(wb_cyc_i) |=> (wb_dat_o == 8'hzz));
assert_a371: assert property(@(posedge wb_clk_i or posedge arst_i) if (arst_i) (wb_dat_o == 8'h00) until_with (arst_i == 0 && !wb_we_i && wb_stb_i));
assert_a372: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && $rose(wb_cyc_i)) |-> ##[1:2] ($isunknown(wb_dat_o) == 0));
assert_a373: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && wb_cyc_i) |-> ##1 $stable(wb_dat_o) until !(!wb_we_i && wb_stb_i && wb_cyc_i));
assert_a374: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && $changed(wb_adr_i)) |-> ##[0:$] (wb_ack_o) |=> (wb_dat_o == $past(wb_adr_i, 1)));
assert_a375: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($changed(wb_adr_i) && !wb_we_i && wb_stb_i) |-> ##[0:$] (wb_ack_o) |=> (wb_dat_o == $past(wb_adr_i, 1)));
assert_a376: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (irq_flag && ien) |=> wb_inta_o);
assert_a377: assert property(@(posedge wb_clk_i) (ien && irq_flag && $rose(ien && irq_flag)) |-> wb_inta_o);
assert_a378: assert property(@(posedge wb_clk_i) (!rst_i && !wb_rst_i && irq_flag && ien) |=> wb_inta_o);
assert_a379: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) !ien |=> !wb_inta_o);
assert_a380: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |=> !wb_inta_o);
assert_a381: assert property(@(posedge wb_clk_i) !rst_i |=> !wb_inta_o);
assert_a382: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($stable(irq_flag) && $stable(ien) && irq_flag && ien) |-> $stable(wb_inta_o));
assert_a383: assert property(@(posedge wb_clk_i) disable iff (!rst_i || wb_rst_i) !wb_inta_o);
assert_a384: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($fell(irq_flag) || $fell(ien)) |=> !wb_inta_o);
assert_a385: assert property(@(posedge wb_clk_i) disable iff (!rst_i || wb_rst_i) (irq_flag && ien) |=> wb_inta_o);
assert_a386: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($changed(irq_flag) || $changed(ien)) |=> (wb_inta_o == (irq_flag && ien)));
assert_a387: assert property(@(posedge wb_clk_i) $fell(ien) |=> !wb_inta_o);
assert_a388: assert property(@(posedge wb_clk_i) disable iff (!rst_i || wb_rst_i) (ien && irq_flag) |=> wb_inta_o);
assert_a389: assert property(@(posedge wb_clk_i) $rose(rst_i) |=> !wb_inta_o);
assert_a390: assert property(@(posedge wb_clk_i) $stable(wb_rst_i) && $stable(irq_flag) && $stable(ien) |-> $stable(wb_inta_o));
assert_a391: assert property(@(posedge wb_clk_i) $rose(!wb_rst_i) |-> ##[0:2] !wb_inta_o);
assert_a392: assert property(@(posedge wb_clk_i) !irq_flag |-> !wb_inta_o);
assert_a393: assert property(@(posedge wb_clk_i) (rst_i || wb_rst_i) |-> !wb_inta_o);
assert_a394: assert property(@(posedge wb_clk_i) $past($stable(irq_flag) && $stable(ien)) |-> $stable(wb_inta_o));
assert_a395: assert property(@(posedge wb_clk_i) disable iff (rst_i || wb_rst_i) !ien |=> !wb_inta_o);
assert_a396: assert property(@(posedge wb_clk_i) $stable(ien) && $stable(irq_flag) |-> $stable(wb_inta_o));
assert_a397: assert property(@(posedge wb_clk_i) wb_rst_i |=> !wb_inta_o);
assert_a398: assert property(@(posedge wb_clk_i) wb_rst_i |-> (wb_ack_o == 1'b0));
assert_a399: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |-> ##[1:2] (wb_cyc_i && wb_stb_i) |-> (wb_ack_o == 1'b1));
assert_a400: assert property(@(posedge wb_clk_i) $rose(wb_rst_i) |=> (wb_dat_o == 8'h0 && wb_ack_o == 1'b0 && wb_inta_o == 1'b0 && scl_pad_o == 1'b0 && scl_padoen_o == 1'b1 && sda_pad_o == 1'b0 && sda_padoen_o == 1'b1));
assert_a401: assert property(@(posedge wb_clk_i) (!$arst_i && $fell(wb_rst_i)) |-> ##[1:4] (wb_cyc_i && wb_stb_i) |-> (wb_ack_o == 1'b1));
assert_a402: assert property(@(posedge wb_clk_i) (wb_we_i && wb_stb_i && wb_cyc_i && $rose(wb_rst_i)) |-> ##1 $stable(wb_dat_o));
assert_a403: assert property(@(posedge wb_clk_i) (!$wb_we_i && wb_stb_i && wb_cyc_i && $rose(wb_rst_i)) |-> (wb_dat_o == 8'h0 && wb_ack_o == 1'b0));
assert_a404: assert property(@(posedge wb_clk_i) (wb_we_i && wb_stb_i && wb_cyc_i && $rose(wb_rst_i)) |=> $stable(wb_dat_o));
assert_a405: assert property(@(posedge wb_clk_i) (wb_cyc_i && wb_stb_i && $rose(wb_rst_i)) |-> (wb_ack_o == 1'b0) and ##1 (!wb_cyc_i && !wb_stb_i));
assert_a406: assert property(@(posedge wb_clk_i) wb_rst_i |-> $stable({wb_adr_i, wb_dat_i, wb_we_i, wb_stb_i, wb_cyc_i, scl_pad_i, sda_pad_i}));
assert_a407: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |-> ##[1:3] (sda_padoen_o == 1'b1 && scl_padoen_o == 1'b1));
assert_a408: assert property(@(posedge wb_clk_i) wb_rst_i |=> (wb_dat_o == 8'h0));
assert_a409: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |-> ##[1:2] (wb_inta_o == $past(wb_inta_o, 2) || wb_inta_o == 1'b1));
assert_a410: assert property(@(posedge wb_clk_i) (wb_rst_i && arst_i) |=> !$past(arst_i));
assert_a411: assert property(@(posedge wb_clk_i) (!$rose(wb_rst_i) && !$rose(arst_i)) |-> ##[1:2] $stable({scl_pad_o, scl_padoen_o, sda_pad_o, sda_padoen_o}));
assert_a412: assert property(@(posedge wb_clk_i) wb_rst_i |=> (wb_dat_o == 8'h0 && !wb_ack_o && !wb_inta_o && !scl_pad_o && scl_padoen_o && !sda_pad_o && sda_padoen_o));
assert_a413: assert property(@(posedge wb_clk_i) arst_i |-> (wb_dat_o == 8'h0 && !wb_ack_o && !wb_inta_o && !scl_pad_o && scl_padoen_o && !sda_pad_o && sda_padoen_o));
assert_a414: assert property(@(posedge wb_clk_i) (wb_cyc_i && $rose(wb_rst_i)) |-> !wb_ack_o until !wb_rst_i);
assert_a415: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |-> ##[1:3] (scl_pad_o == 1'b1 && scl_padoen_o == 1'b1 && sda_pad_o == 1'b1 && sda_padoen_o == 1'b1));
assert_a416: assert property(@(posedge wb_clk_i) (wb_rst_i && wb_cyc_i) |=> !wb_ack_o);
assert_a417: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && wb_stb_i && !wb_rst_i) |=> !$stable(wb_dat_o));
assert_a418: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($fell(wb_rst_i) ##1 wb_we_i && wb_stb_i && wb_cyc_i) |-> ##[1:3] wb_ack_o);
assert_a419: assert property(@(posedge wb_clk_i) (wb_rst_i) |-> (scl_padoen_o && sda_padoen_o));
assert_a420: assert property(@(posedge wb_clk_i) $rose(wb_rst_i) |-> wb_rst_i [*2]);
assert_a421: assert property(@(posedge wb_clk_i) (!wb_rst_i && !arst_i) |-> ##[1:2] (wb_stb_i && wb_cyc_i) |-> wb_ack_o);
assert_a422: assert property(@(posedge wb_clk_i) $rose(wb_rst_i) |=> (wb_dat_o == '0 && wb_ack_o == '0 && wb_inta_o == '0 && scl_pad_o == '0 && scl_padoen_o == '1 && sda_pad_o == '0 && sda_padoen_o == '1));
assert_a423: assert property(@(posedge wb_clk_i) $rose(wb_rst_i) |=> (wb_adr_i == '0) ##1 (wb_stb_i && wb_cyc_i && !wb_we_i) |-> (wb_dat_o == '0));
assert_a424: assert property(@(posedge wb_clk_i) (wb_rst_i || arst_i) |-> !wb_inta_o);
assert_a425: assert property(@(posedge wb_clk_i) (wb_rst_i && wb_inta_o) |=> !wb_inta_o);
assert_a426: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |=> $stable(wb_dat_o) && $stable(wb_ack_o) && $stable(wb_inta_o));
assert_a427: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_we_i) |-> (wb_ack_o || $stable(wb_dat_o)));
assert_a428: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (!wb_stb_i [*5]) |-> $stable(wb_dat_o)[*5]);
assert_a429: assert property(@(posedge wb_clk_i) (wb_rst_i && wb_stb_i) |=> !wb_ack_o);
assert_a430: assert property(@(posedge wb_clk_i) (arst_i && !wb_stb_i) |-> !wb_inta_o);
assert_a431: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && !wb_cyc_i) |=> !wb_ack_o);
assert_a432: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_cyc_i) |=> wb_ack_o);
assert_a433: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) $rose(wb_stb_i) |-> ##1 wb_stb_i[*1:$] ##1 wb_ack_o);
assert_a434: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_we_i && wb_cyc_i) |-> ##1 (wb_ack_o |-> $stable(wb_dat_i)));
assert_a435: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (!wb_stb_i [*10]) |-> !wb_inta_o[*10]);
assert_a436: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && !wb_cyc_i) |-> !wb_ack_o);
assert_a437: assert property(@(posedge wb_clk_i) disable iff (!arst_i) $rose(wb_stb_i) || $fell(wb_stb_i) |=> $stable(wb_ack_o));
assert_a438: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_cyc_i && wb_ack_o) |=> !wb_ack_o until !wb_stb_i);
assert_a439: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (!wb_stb_i && wb_cyc_i) |=> !wb_ack_o);
assert_a440: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) $changed(wb_we_i) |-> $onehot0({$past(wb_stb_i), wb_stb_i}));
assert_a441: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) ($fell(wb_stb_i) && wb_cyc_i) |=> !wb_ack_o);
assert_a442: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) ($rose(wb_stb_i) && wb_we_i) |-> $stable(wb_dat_o) until !wb_stb_i);
assert_a443: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_adr_i inside {3'b000,3'b001,3'b010,3'b011}) |-> ##[1:2] (wb_ack_o && $stable(wb_dat_o)[*1:$] ##0 !$stable(wb_dat_o)));
assert_a444: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (arst_i && wb_stb_i) |=> !wb_ack_o);
assert_a445: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_cyc_i && $rose(wb_stb_i)) |-> ##[0:1] wb_ack_o);
assert_a446: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_cyc_i) |-> ##[1:3] wb_ack_o);
assert_a447: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i || arst_i) (wb_stb_i && wb_adr_i inside {3'b000,3'b001,3'b010,3'b011}) |-> $stable(wb_dat_o) until wb_ack_o);
assert_a448: assert property(@(posedge wb_clk_i) (wb_stb_i && !wb_we_i) |-> ##[1:4] $isunknown(wb_dat_o) == 0);
assert_a449: assert property(@(posedge wb_clk_i) (wb_cyc_i && $past(wb_stb_i) && !wb_stb_i) |-> !wb_ack_o);
assert_a450: assert property(@(posedge wb_clk_i) (!wb_rst_i && !wb_stb_i) |-> !wb_ack_o);
assert_a451: assert property(@(posedge wb_clk_i) (wb_cyc_i && !wb_stb_i) |-> $stable(wb_inta_o));
assert_a452: assert property(@(posedge wb_clk_i) (wb_rst_i && wb_stb_i) |-> !wb_ack_o);
assert_a453: assert property(@(posedge wb_clk_i) $fell(wb_rst_i) |-> ##[1:2] !wb_stb_i);
assert_a454: assert property(@(posedge wb_clk_i) (wb_cyc_i && !wb_stb_i) |-> !wb_ack_o);
assert_a455: assert property(@(posedge wb_clk_i) (arst_i && !wb_stb_i) |-> !wb_ack_o);
assert_a456: assert property(@(posedge wb_clk_i) (wb_stb_i && !wb_ack_o) |-> ##[1:8] wb_ack_o);
assert_a457: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && !wb_rst_i) |-> ##[1:2] ($isunknown(wb_dat_o) == 0));
assert_a458: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($changed(wb_adr_i) && wb_we_i && wb_stb_i) |=> (wb_dat_i == $past(wb_dat_i)));
assert_a459: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i && wb_ack_o) |=> $stable(wb_dat_o)[*1]);
assert_a460: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_we_i && $changed(wb_adr_i)) |-> !wb_ack_o until $stable(wb_adr_i)[*1]);
assert_a461: assert property(@(posedge wb_clk_i) (arst_i && wb_we_i) |-> (wb_dat_o == 8'h00) until_with (arst_i));
assert_a462: assert property(@(posedge wb_clk_i) (!wb_cyc_i) |-> !wb_we_i throughout !wb_cyc_i);
assert_a463: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_we_i && wb_stb_i && wb_cyc_i)[*3] |-> (wb_ack_o [->1]));
assert_a464: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $rose(wb_stb_i) |-> ($stable(wb_we_i) && $stable(wb_dat_i))[*-1]);
assert_a465: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_we_i && wb_stb_i) |=> $stable(wb_dat_o) throughout (wb_ack_o [->1]));
assert_a466: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_stb_i) |-> ($stable(wb_dat_o) && $stable(wb_dat_i)));
assert_a467: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_ack_o) |-> (wb_dat_o inside {VALID_READ_VALUES}));
assert_a468: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_cyc_i) |-> (wb_dat_o == $past(wb_dat_o)) until wb_ack_o);
assert_a469: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $changed(wb_we_i) |-> (wb_cyc_i && wb_stb_i));
assert_a470: assert property(@(posedge wb_clk_i) disable iff (!arst_i) (wb_we_i && arst_i) |-> (wb_dat_o == RESET_VALUE));
assert_a471: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) ($fell(wb_we_i) && wb_cyc_i) |-> !wb_inta_o);
assert_a472: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_cyc_i && $changed(wb_we_i)) |-> $stable(wb_dat_o) until !wb_cyc_i);
assert_a473: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) $fell(wb_stb_i) |-> $stable(wb_we_i)[*1]);
assert_a474: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (!wb_we_i && wb_cyc_i) |-> $stable(wb_dat_o));
assert_a475: assert property(@(posedge wb_clk_i) disable iff (wb_rst_i) (wb_we_i && wb_inta_o) |-> (wb_dat_o == INTERRUPT_DATA));
assert_a476: assert property(@(posedge wb_clk_i) ($rose(wb_we_i) && wb_stb_i) |-> ##[1:3] wb_ack_o);
assert_a477: assert property(@(posedge wb_clk_i) (wb_we_i && $changed(wb_adr_i)) |-> $stable(wb_dat_o) until_within ($fell(wb_we_i) && $rose(wb_we_i)));
assert_a478: assert property(@(posedge wb_clk_i) ($fell(wb_we_i) && wb_cyc_i) |-> $stable(wb_dat_o) until_within $rose(wb_cyc_i));
assert_a479: assert property(@(posedge wb_clk_i) arst_i |-> !wb_we_i);
assert_a480: assert property(@(posedge wb_clk_i) (wb_we_i && wb_rst_i) |-> ##1 (wb_dat_o === 8'h00));
assert_a481: assert property(@(posedge wb_clk_i) (!wb_we_i && wb_cyc_i) |-> !wb_inta_o);
assert_a482: assert property(@(posedge wb_clk_i) (wb_rst_i && $changed(wb_we_i)) |-> $stable(wb_dat_o));
assert_a483: assert property(@(posedge wb_clk_i) (wb_we_i && wb_rst_i) |-> (wb_dat_o === 8'h00) || $stable(wb_dat_o) until_within !wb_rst_i);
assert_a484: assert property(@(posedge wb_clk_i) (wb_we_i && wb_stb_i && $rose(wb_cyc_i)) |-> ##[1:2] wb_ack_o);
assert_a485: assert property(@(posedge wb_clk_i) (arst_i && $changed(wb_we_i)) |-> $stable(wb_dat_o) until_within !arst_i);

endmodule
