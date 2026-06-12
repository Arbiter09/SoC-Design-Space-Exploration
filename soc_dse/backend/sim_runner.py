"""
Verilator simulation runner.

Compiles each config's RTL with Verilator, links the shared testbench,
runs the simulation, and parses the output for cycle count, assertion
violations, and toggle coverage.

Environment variables
---------------------
VERILATOR_BIN   Path to the verilator binary (default: ``verilator``)
SOC_CONFIGS_DIR Root directory for generated configs (default: soc_dse/configs/)
SOC_TESTBENCH   Path to testbench.cpp (default: soc_dse/testbench/testbench.cpp)
SOC_DB_PATH     SQLite database path (default: soc_dse/dse.db)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from soc_dse.backend import db

log = logging.getLogger(__name__)

VERILATOR_BIN: str = os.environ.get("VERILATOR_BIN", "verilator")
_ROOT = Path(__file__).parent.parent
CONFIGS_DIR: Path = Path(os.environ.get("SOC_CONFIGS_DIR", _ROOT / "configs"))
TESTBENCH: Path = Path(
    os.environ.get("SOC_TESTBENCH", _ROOT / "testbench" / "testbench.cpp")
)

# ---------------------------------------------------------------------------
# Compile step
# ---------------------------------------------------------------------------

def _compile_config(config_id: str, config_dir: Path) -> Optional[Path]:
    """
    Invoke Verilator to compile RTL + testbench into a C++ simulation binary.

    Returns the path to the compiled binary, or None on failure.
    """
    rtl_dir = config_dir / "rtl"
    obj_dir = config_dir / "obj_dir"
    compile_log = config_dir / "verilator_compile.log"

    cmd = [
        VERILATOR_BIN,
        "--cc",
        str(rtl_dir / "soc_top.v"),
        "--exe", str(TESTBENCH),
        "--Mdir", str(obj_dir),
        "--build",
        "--top-module", "soc_top",
        "-Wall",
        "--jobs", "4",
        "-CFLAGS", "-O2",
    ]

    log.info("[%s] Compiling with Verilator …", config_id)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - t0
    compile_log.write_text(result.stdout + "\n" + result.stderr)

    if result.returncode != 0:
        log.warning(
            "[%s] Verilator compile failed (%.1fs): %s",
            config_id, elapsed, result.stderr[:200],
        )
        return None

    binary = obj_dir / "Vsoc_top"
    if not binary.exists():
        log.warning("[%s] Expected binary not found: %s", config_id, binary)
        return None

    log.info("[%s] Compile OK (%.1fs)", config_id, elapsed)
    return binary


# ---------------------------------------------------------------------------
# Parse simulation output
# ---------------------------------------------------------------------------

_RE_CYCLES = re.compile(r"CYCLES:\s*(\d+)", re.IGNORECASE)
_RE_ASSERTIONS = re.compile(r"ASSERTION_FAILURES:\s*(\d+)", re.IGNORECASE)
_RE_COVERAGE = re.compile(r"TOGGLE_COVERAGE:\s*([\d.]+)", re.IGNORECASE)


def _parse_sim_output(stdout: str) -> dict[str, int | float]:
    """
    Extract metrics from simulation stdout.

    The testbench emits lines like::

        CYCLES: 1024
        ASSERTION_FAILURES: 0
        TOGGLE_COVERAGE: 0.87
    """
    cycle_m = _RE_CYCLES.search(stdout)
    assert_m = _RE_ASSERTIONS.search(stdout)
    cov_m = _RE_COVERAGE.search(stdout)

    return {
        "cycle_count": int(cycle_m.group(1)) if cycle_m else -1,
        "assertion_violations": int(assert_m.group(1)) if assert_m else -1,
        "toggle_coverage": float(cov_m.group(1)) if cov_m else -1.0,
    }


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(config_id: str) -> bool:
    """
    Compile and run the Verilator simulation for a single config.

    Returns True on success, False on failure.
    """
    config_dir = CONFIGS_DIR / config_id

    if not config_dir.exists():
        log.error("Config directory not found: %s", config_dir)
        db.upsert_simulation(config_id, status="error", error_msg="config_dir missing")
        return False

    vcd_path = config_dir / "sim.vcd"

    env = os.environ.copy()
    env["SIM_VCD_PATH"] = str(vcd_path)
    env["SIM_CONFIG_ID"] = config_id

    try:
        binary = _compile_config(config_id, config_dir)
        if binary is None:
            db.upsert_simulation(config_id, status="error", error_msg="compile failed")
            return False

        log.info("[%s] Running simulation …", config_id)
        t0 = time.perf_counter()
        result = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        runtime_s = time.perf_counter() - t0
        (config_dir / "sim.log").write_text(result.stdout + "\n" + result.stderr)

        if result.returncode != 0:
            log.warning("[%s] Simulation exited with code %d", config_id, result.returncode)
            db.upsert_simulation(
                config_id,
                runtime_s=round(runtime_s, 3),
                status="error",
                error_msg=f"sim exit {result.returncode}",
            )
            return False

        metrics = _parse_sim_output(result.stdout)
        db.upsert_simulation(
            config_id,
            runtime_s=round(runtime_s, 3),
            status="ok",
            **metrics,  # type: ignore[arg-type]
        )
        log.info(
            "[%s] Simulation OK  cycles=%d assertions=%d coverage=%.2f  (%.1fs)",
            config_id,
            metrics["cycle_count"],
            metrics["assertion_violations"],
            metrics["toggle_coverage"],
            runtime_s,
        )
        return True

    except subprocess.TimeoutExpired:
        log.error("[%s] Simulation timed out", config_id)
        db.upsert_simulation(config_id, status="timeout", error_msg="subprocess timeout")
        return False
    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] Unexpected error: %s", config_id, exc)
        db.upsert_simulation(config_id, status="error", error_msg=str(exc))
        return False


def run_simulation_batch(config_ids: list[str], *, skip_done: bool = True) -> dict[str, bool]:
    """Run simulation for a list of config IDs; return {config_id: success}."""
    results: dict[str, bool] = {}
    for cid in config_ids:
        if skip_done and db.simulation_done(cid):
            log.debug("[%s] Simulation already done, skipping", cid)
            results[cid] = True
            continue
        results[cid] = run_simulation(cid)
    return results
