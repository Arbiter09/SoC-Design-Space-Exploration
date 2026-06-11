"""
SQLite schema initialisation and query helpers for the SoC DSE platform.

All table creation is idempotent (CREATE TABLE IF NOT EXISTS).
The database path is read from the environment variable ``SOC_DB_PATH``
and defaults to ``soc_dse/dse.db``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

_DEFAULT_DB_PATH = Path(__file__).parent.parent / "dse.db"
DB_PATH: Path = Path(os.environ.get("SOC_DB_PATH", _DEFAULT_DB_PATH))

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS configs (
    config_id   TEXT PRIMARY KEY,
    params_json TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS synthesis_results (
    config_id       TEXT PRIMARY KEY REFERENCES configs(config_id),
    cell_count      INTEGER,
    wire_count      INTEGER,
    area            REAL,
    delay           REAL,
    power_estimate  REAL,
    runtime_s       REAL,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_msg       TEXT,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS simulation_results (
    config_id          TEXT PRIMARY KEY REFERENCES configs(config_id),
    cycle_count        INTEGER,
    assertion_violations INTEGER,
    toggle_coverage    REAL,
    runtime_s          REAL,
    status             TEXT NOT NULL DEFAULT 'pending',
    error_msg          TEXT,
    updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS triage_results (
    config_id              TEXT PRIMARY KEY REFERENCES configs(config_id),
    failure_mode           TEXT,
    affected_signals_json  TEXT,
    fix_hint               TEXT,
    updated_at             TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def init_db(path: Optional[Path] = None) -> None:
    """Create all tables if they do not already exist."""
    db = path or DB_PATH
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(_DDL)
        conn.commit()


@contextmanager
def get_conn(path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with row_factory set to :class:`sqlite3.Row`."""
    db = path or DB_PATH
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Config table helpers
# ---------------------------------------------------------------------------

def insert_config(config_id: str, params: dict[str, Any]) -> None:
    """Insert a config row; silently skip if it already exists."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO configs (config_id, params_json) VALUES (?, ?)",
            (config_id, json.dumps(params)),
        )


def config_exists(config_id: str) -> bool:
    """Return True if *config_id* is present in the configs table."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM configs WHERE config_id = ?", (config_id,)
        ).fetchone()
    return row is not None


def get_all_config_ids() -> list[str]:
    """Return all known config IDs."""
    with get_conn() as conn:
        rows = conn.execute("SELECT config_id FROM configs").fetchall()
    return [r["config_id"] for r in rows]


# ---------------------------------------------------------------------------
# Synthesis results helpers
# ---------------------------------------------------------------------------

def upsert_synthesis(
    config_id: str,
    *,
    cell_count: Optional[int] = None,
    wire_count: Optional[int] = None,
    area: Optional[float] = None,
    delay: Optional[float] = None,
    power_estimate: Optional[float] = None,
    runtime_s: Optional[float] = None,
    status: str = "ok",
    error_msg: Optional[str] = None,
) -> None:
    """Insert or replace a synthesis result row."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO synthesis_results
                (config_id, cell_count, wire_count, area, delay,
                 power_estimate, runtime_s, status, error_msg, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(config_id) DO UPDATE SET
                cell_count      = excluded.cell_count,
                wire_count      = excluded.wire_count,
                area            = excluded.area,
                delay           = excluded.delay,
                power_estimate  = excluded.power_estimate,
                runtime_s       = excluded.runtime_s,
                status          = excluded.status,
                error_msg       = excluded.error_msg,
                updated_at      = excluded.updated_at
            """,
            (config_id, cell_count, wire_count, area, delay,
             power_estimate, runtime_s, status, error_msg),
        )


def synthesis_done(config_id: str) -> bool:
    """Return True if synthesis completed successfully for this config."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM synthesis_results WHERE config_id = ?",
            (config_id,),
        ).fetchone()
    return row is not None and row["status"] == "ok"


def get_all_synthesis_results() -> list[dict[str, Any]]:
    """Return all successful synthesis results as a list of dicts."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM synthesis_results WHERE status = 'ok'"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Simulation results helpers
# ---------------------------------------------------------------------------

def upsert_simulation(
    config_id: str,
    *,
    cycle_count: Optional[int] = None,
    assertion_violations: Optional[int] = None,
    toggle_coverage: Optional[float] = None,
    runtime_s: Optional[float] = None,
    status: str = "ok",
    error_msg: Optional[str] = None,
) -> None:
    """Insert or replace a simulation result row."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO simulation_results
                (config_id, cycle_count, assertion_violations, toggle_coverage,
                 runtime_s, status, error_msg, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(config_id) DO UPDATE SET
                cycle_count          = excluded.cycle_count,
                assertion_violations = excluded.assertion_violations,
                toggle_coverage      = excluded.toggle_coverage,
                runtime_s            = excluded.runtime_s,
                status               = excluded.status,
                error_msg            = excluded.error_msg,
                updated_at           = excluded.updated_at
            """,
            (config_id, cycle_count, assertion_violations, toggle_coverage,
             runtime_s, status, error_msg),
        )


def simulation_done(config_id: str) -> bool:
    """Return True if simulation completed successfully for this config."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM simulation_results WHERE config_id = ?",
            (config_id,),
        ).fetchone()
    return row is not None and row["status"] == "ok"


# ---------------------------------------------------------------------------
# Triage results helpers
# ---------------------------------------------------------------------------

def upsert_triage(
    config_id: str,
    *,
    failure_mode: str,
    affected_signals: list[str],
    fix_hint: str,
) -> None:
    """Insert or replace a triage result row."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO triage_results
                (config_id, failure_mode, affected_signals_json, fix_hint, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(config_id) DO UPDATE SET
                failure_mode          = excluded.failure_mode,
                affected_signals_json = excluded.affected_signals_json,
                fix_hint              = excluded.fix_hint,
                updated_at            = excluded.updated_at
            """,
            (config_id, failure_mode, json.dumps(affected_signals), fix_hint),
        )
