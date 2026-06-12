"""
Yosys synthesis runner.

For each config, generates a ``.ys`` script, invokes Yosys as a subprocess,
and extracts PPA metrics **from the Yosys JSON netlist** (stable across
Yosys versions) rather than parsing stdout.

Environment variables
---------------------
YOSYS_BIN       Path to the yosys binary (default: ``yosys``)
SOC_CONFIGS_DIR Root directory for generated configs (default: soc_dse/configs/)
SOC_DB_PATH     SQLite database path (default: soc_dse/dse.db)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from soc_dse.backend import db

log = logging.getLogger(__name__)

YOSYS_BIN: str = os.environ.get("YOSYS_BIN", "yosys")
_ROOT = Path(__file__).parent.parent
CONFIGS_DIR: Path = Path(os.environ.get("SOC_CONFIGS_DIR", _ROOT / "configs"))

# Yosys cell types that contribute to "power" estimation (rough proxy)
_SEQUENTIAL_CELLS = {"$dff", "$dffe", "$dffsr", "$dffsre", "$sdff", "$sdffe"}

# ---------------------------------------------------------------------------
# Yosys script generation
# ---------------------------------------------------------------------------

_YS_TEMPLATE = """\
read_verilog {rtl_dir}/pipeline.v
read_verilog {rtl_dir}/cache.v
read_verilog {rtl_dir}/soc_top.v
hierarchy -check -top soc_top
proc; opt; fsm; opt; memory; opt
synth -top soc_top
stat
write_json {netlist_json}
"""


def _write_ys_script(config_dir: Path) -> tuple[Path, Path]:
    """Write the .ys synthesis script and return (script_path, netlist_json_path)."""
    rtl_dir = config_dir / "rtl"
    script_path = config_dir / "synth.ys"
    netlist_path = config_dir / "netlist.json"

    script_path.write_text(
        _YS_TEMPLATE.format(
            rtl_dir=rtl_dir,
            netlist_json=netlist_path,
        )
    )
    return script_path, netlist_path


# ---------------------------------------------------------------------------
# PPA extraction from netlist JSON
# ---------------------------------------------------------------------------

def _extract_ppa_from_netlist(netlist_path: Path) -> dict[str, float | int]:
    """
    Parse the Yosys JSON netlist to extract PPA metrics.

    Uses the structured JSON output rather than stdout regex — this is stable
    across all Yosys versions.

    Returns a dict with keys: cell_count, wire_count, power_estimate, area, delay.
    """
    with netlist_path.open() as fh:
        netlist: dict = json.load(fh)

    modules: dict = netlist.get("modules", {})

    # Find the top module (prefer 'soc_top', fall back to first)
    top_name = "soc_top" if "soc_top" in modules else next(iter(modules), None)
    if top_name is None:
        raise ValueError("No modules found in Yosys netlist JSON")

    top = modules[top_name]
    cells: dict = top.get("cells", {})
    netnames: dict = top.get("netnames", {})

    cell_count = len(cells)
    wire_count = len(netnames)

    # Rough area proxy: each cell contributes 1.0 unit; sequential cells cost 2.0
    area: float = sum(
        2.0 if cell_info.get("type", "").lower() in _SEQUENTIAL_CELLS else 1.0
        for cell_info in cells.values()
    )

    # Power proxy: proportion of sequential cells × area
    seq_count = sum(
        1 for cell_info in cells.values()
        if cell_info.get("type", "").lower() in _SEQUENTIAL_CELLS
    )
    power_estimate: float = round((seq_count / max(cell_count, 1)) * area, 4)

    # Delay proxy: average fan-in depth (number of bits in connections)
    total_bits = sum(
        len(port_bits)
        for cell_info in cells.values()
        for port_bits in cell_info.get("connections", {}).values()
    )
    delay: float = round(total_bits / max(cell_count, 1), 4)

    return {
        "cell_count": cell_count,
        "wire_count": wire_count,
        "area": round(area, 4),
        "delay": delay,
        "power_estimate": power_estimate,
    }


# ---------------------------------------------------------------------------
# Synthesis runner
# ---------------------------------------------------------------------------

def run_synthesis(config_id: str) -> bool:
    """
    Run Yosys synthesis for a single config.

    Returns True on success, False on failure.
    Writes results to the ``synthesis_results`` table.
    """
    config_dir = CONFIGS_DIR / config_id

    if not config_dir.exists():
        log.error("Config directory not found: %s", config_dir)
        db.upsert_synthesis(config_id, status="error", error_msg="config_dir missing")
        return False

    script_path, netlist_path = _write_ys_script(config_dir)
    log_path = config_dir / "synth.log"

    log.info("[%s] Starting Yosys synthesis …", config_id)
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            [YOSYS_BIN, "-s", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        runtime_s = time.perf_counter() - t0
        log_path.write_text(result.stdout + "\n" + result.stderr)

        if result.returncode != 0:
            log.warning("[%s] Yosys exited with code %d", config_id, result.returncode)
            db.upsert_synthesis(
                config_id,
                runtime_s=runtime_s,
                status="error",
                error_msg=f"yosys exit {result.returncode}",
            )
            return False

        if not netlist_path.exists():
            db.upsert_synthesis(
                config_id,
                runtime_s=runtime_s,
                status="error",
                error_msg="netlist.json not produced",
            )
            return False

        metrics = _extract_ppa_from_netlist(netlist_path)
        db.upsert_synthesis(
            config_id,
            runtime_s=round(runtime_s, 3),
            status="ok",
            **metrics,  # type: ignore[arg-type]
        )
        log.info(
            "[%s] Synthesis OK  cells=%d wires=%d area=%.1f delay=%.2f power=%.4f  (%.1fs)",
            config_id,
            metrics["cell_count"],
            metrics["wire_count"],
            metrics["area"],
            metrics["delay"],
            metrics["power_estimate"],
            runtime_s,
        )
        return True

    except subprocess.TimeoutExpired:
        log.error("[%s] Yosys timed out", config_id)
        db.upsert_synthesis(config_id, status="timeout", error_msg="subprocess timeout")
        return False
    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] Unexpected error: %s", config_id, exc)
        db.upsert_synthesis(config_id, status="error", error_msg=str(exc))
        return False


def run_synthesis_batch(config_ids: list[str], *, skip_done: bool = True) -> dict[str, bool]:
    """Run synthesis for a list of config IDs; return {config_id: success}."""
    results: dict[str, bool] = {}
    for cid in config_ids:
        if skip_done and db.synthesis_done(cid):
            log.debug("[%s] Synthesis already done, skipping", cid)
            results[cid] = True
            continue
        results[cid] = run_synthesis(cid)
    return results
