// Verilator testbench for soc_top
//
// Drives the DUT through a fixed set of stimulus patterns, checks basic
// assertions, dumps a VCD waveform, and prints structured metrics to stdout
// that sim_runner.py can parse:
//
//   CYCLES: <n>
//   ASSERTION_FAILURES: <n>
//   TOGGLE_COVERAGE: <0.0–1.0>
//
// Environment variables (set by sim_runner.py):
//   SIM_VCD_PATH   – path to write the VCD file  (default: sim.vcd)
//   SIM_CONFIG_ID  – config ID string for logging (default: unknown)

#include "Vsoc_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
static constexpr uint64_t MAX_CYCLES      = 2048;
static constexpr uint64_t RESET_CYCLES    = 8;
static constexpr uint64_t STIMULUS_PERIOD = 16;   // new stimulus every N cycles

// ---------------------------------------------------------------------------
// Toggle coverage: track signal transitions
// ---------------------------------------------------------------------------
struct ToggleTracker {
    uint64_t toggled = 0;
    uint64_t total   = 0;

    // Track a single 1-bit signal
    void track(uint8_t prev, uint8_t cur) {
        total++;
        if (prev != cur) toggled++;
    }

    double coverage() const {
        return (total == 0) ? 0.0 : static_cast<double>(toggled) / total;
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    // Read environment
    const char* vcd_env = std::getenv("SIM_VCD_PATH");
    std::string vcd_path = vcd_env ? vcd_env : "sim.vcd";

    const char* cfg_env = std::getenv("SIM_CONFIG_ID");
    std::string config_id = cfg_env ? cfg_env : "unknown";

    // Instantiate DUT
    Vsoc_top* dut = new Vsoc_top;

    // VCD trace
    VerilatedVcdC* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open(vcd_path.c_str());

    // ---------------------------------------------------------------------------
    // Simulation loop
    // ---------------------------------------------------------------------------
    uint64_t    sim_time          = 0;
    uint64_t    cycle             = 0;
    int         assertion_failures = 0;
    ToggleTracker tracker;

    // Previous signal values for toggle tracking
    uint8_t prev_pipe_valid = 0;
    uint8_t prev_cache_hit  = 0;
    uint8_t prev_stall      = 0;

    // Reset
    dut->clk    = 0;
    dut->rst_n  = 0;
    dut->cpu_valid      = 0;
    dut->cpu_data       = 0;
    dut->cpu_opcode     = 0;
    dut->mem_req_valid  = 0;
    dut->mem_req_we     = 0;
    dut->mem_req_addr   = 0;
    dut->mem_req_wdata  = 0;

    while (cycle < MAX_CYCLES && !Verilated::gotFinish()) {
        // Clock toggle
        dut->clk = !dut->clk;
        sim_time++;

        if (dut->clk) {  // rising edge
            cycle++;

            // Deassert reset after RESET_CYCLES
            if (cycle == RESET_CYCLES) {
                dut->rst_n = 1;
            }

            // Apply stimulus every STIMULUS_PERIOD cycles
            if (cycle > RESET_CYCLES && (cycle % STIMULUS_PERIOD) == 0) {
                dut->cpu_valid     = 1;
                dut->cpu_data      = static_cast<uint32_t>(cycle * 0x13579BDF);
                dut->cpu_opcode    = static_cast<uint8_t>((cycle / STIMULUS_PERIOD) & 0xF);
                dut->mem_req_valid = 1;
                dut->mem_req_we    = (cycle & 1) ? 1 : 0;
                dut->mem_req_addr  = static_cast<uint32_t>((cycle >> 4) & 0xFF);
                dut->mem_req_wdata = static_cast<uint32_t>(cycle ^ 0xDEADBEEF);
            } else {
                dut->cpu_valid     = 0;
                dut->mem_req_valid = 0;
            }
        }

        // Evaluate
        dut->eval();
        tfp->dump(sim_time);

        // Assertions (rising edge only)
        if (dut->clk && cycle > RESET_CYCLES) {
            // Assert: stall and valid_out should not be simultaneously asserted
            if (dut->pipe_stall && dut->pipe_valid_out) {
                assertion_failures++;
            }
        }

        // Toggle coverage (rising edge)
        if (dut->clk) {
            tracker.track(prev_pipe_valid, dut->pipe_valid_out);
            tracker.track(prev_cache_hit,  dut->cache_resp_hit);
            tracker.track(prev_stall,      dut->pipe_stall);
            prev_pipe_valid = dut->pipe_valid_out;
            prev_cache_hit  = dut->cache_resp_hit;
            prev_stall      = dut->pipe_stall;
        }
    }

    // Finalise
    dut->final();
    tfp->close();

    // Print structured metrics for sim_runner.py
    std::cout << "CYCLES: "             << cycle                          << "\n";
    std::cout << "ASSERTION_FAILURES: " << assertion_failures             << "\n";
    std::cout << "TOGGLE_COVERAGE: "    << tracker.coverage()             << "\n";

    delete tfp;
    delete dut;
    return (assertion_failures > 0) ? 1 : 0;
}
