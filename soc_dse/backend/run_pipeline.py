"""
Pipeline orchestrator.

Generates RTL configs, then runs Yosys synthesis and Verilator simulation
in parallel using ``concurrent.futures.ProcessPoolExecutor`` with an explicit
``spawn`` start method (avoids deadlocks when PyTorch Geometric is imported
later in the same process tree).

Entry point is guarded with ``if __name__ == "__main__"`` as required by
the ``spawn`` start method on all platforms.

Usage
-----
    python -m soc_dse.backend.run_pipeline [options]

Options
-------
--param-space PATH   Path to param_space.yaml
--sample N           Limit to N randomly sampled configs
--seed INT           RNG seed (default: 42)
--synth-only         Skip simulation
--sim-only           Skip synthesis (requires existing RTL)
--workers N          Max parallel worker processes (default: CPU count)
--no-skip            Re-run configs already in the DB
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from soc_dse.backend import db
from soc_dse.backend.generator import CONFIGS_DIR, generate_all
from soc_dse.backend.synth_runner import run_synthesis
from soc_dse.backend.sim_runner import run_simulation

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_DEFAULT_PARAM_SPACE = _ROOT / "configs" / "param_space.yaml"


# ---------------------------------------------------------------------------
# Parallel execution helpers
# ---------------------------------------------------------------------------

def _run_parallel(
    fn: Callable[[str], bool],
    config_ids: list[str],
    *,
    max_workers: int,
    stage_name: str,
) -> dict[str, bool]:
    """
    Execute *fn(config_id)* for every ID in *config_ids* using a
    ProcessPoolExecutor with the ``spawn`` start method.

    Returns a dict mapping config_id → success.
    """
    results: dict[str, bool] = {}
    if not config_ids:
        return results

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        future_to_id = {pool.submit(fn, cid): cid for cid in config_ids}
        done = 0
        total = len(config_ids)
        for future in as_completed(future_to_id):
            cid = future_to_id[future]
            done += 1
            try:
                ok = future.result()
            except Exception as exc:  # noqa: BLE001
                log.error("[%s] %s raised: %s", cid, stage_name, exc)
                ok = False
            results[cid] = ok
            log.info("[%s/%s] %s %s: %s", done, total, stage_name, cid, "OK" if ok else "FAIL")

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    param_space_yaml: Path,
    *,
    sample: int | None = None,
    seed: int = 42,
    synth_only: bool = False,
    sim_only: bool = False,
    max_workers: int | None = None,
    skip_existing: bool = True,
) -> None:
    """
    Full DSE pipeline:

    1. Generate all RTL configs (or load existing ones from DB).
    2. Run Yosys synthesis in parallel.
    3. Run Verilator simulation in parallel.
    """
    workers = max_workers or os.cpu_count() or 4
    db.init_db()

    # ---- Step 1: RTL generation ------------------------------------------------
    if not sim_only:
        log.info("=== Generating RTL configs ===")
        new_ids = generate_all(
            param_space_yaml,
            sample=sample,
            seed=seed,
            skip_existing=skip_existing,
        )
        log.info("New configs generated: %d", len(new_ids))

    all_ids = db.get_all_config_ids()
    if not all_ids:
        log.error("No configs found. Run without --sim-only first.")
        sys.exit(1)

    log.info("Total configs in DB: %d", len(all_ids))

    # ---- Step 2: Synthesis -----------------------------------------------------
    if not sim_only:
        log.info("=== Running Yosys synthesis  (workers=%d) ===", workers)
        t0 = time.perf_counter()
        synth_ids = (
            [cid for cid in all_ids if not db.synthesis_done(cid)]
            if skip_existing else all_ids
        )
        synth_results = _run_parallel(
            run_synthesis, synth_ids, max_workers=workers, stage_name="synth"
        )
        elapsed = time.perf_counter() - t0
        n_ok = sum(synth_results.values())
        log.info("Synthesis complete: %d/%d OK  (%.1fs)", n_ok, len(synth_ids), elapsed)

    # ---- Step 3: Simulation ----------------------------------------------------
    if not synth_only:
        log.info("=== Running Verilator simulation  (workers=%d) ===", workers)
        t0 = time.perf_counter()
        sim_ids = (
            [cid for cid in all_ids if not db.simulation_done(cid)]
            if skip_existing else all_ids
        )
        sim_results = _run_parallel(
            run_simulation, sim_ids, max_workers=workers, stage_name="sim"
        )
        elapsed = time.perf_counter() - t0
        n_ok = sum(sim_results.values())
        log.info("Simulation complete: %d/%d OK  (%.1fs)", n_ok, len(sim_ids), elapsed)

    log.info("=== Pipeline finished ===")


# ---------------------------------------------------------------------------
# CLI — must be guarded with __name__ == "__main__" for spawn start method
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SoC DSE full pipeline orchestrator")
    p.add_argument(
        "--param-space",
        type=Path,
        default=_DEFAULT_PARAM_SPACE,
        help="Path to param_space.yaml",
    )
    p.add_argument("--sample", type=int, default=None, help="Limit configs to N random samples")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    p.add_argument("--synth-only", action="store_true", help="Skip simulation")
    p.add_argument("--sim-only", action="store_true", help="Skip RTL generation and synthesis")
    p.add_argument("--workers", type=int, default=None, help="Parallel worker processes")
    p.add_argument("--no-skip", action="store_true", help="Re-run already-completed configs")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if _args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline(
        _args.param_space,
        sample=_args.sample,
        seed=_args.seed,
        synth_only=_args.synth_only,
        sim_only=_args.sim_only,
        max_workers=_args.workers,
        skip_existing=not _args.no_skip,
    )
