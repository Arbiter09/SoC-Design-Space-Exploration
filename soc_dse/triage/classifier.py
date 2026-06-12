"""
RTL failure mode classifier.

Two-stage classification pipeline:

Stage 1 — Rule-based pass
    Regex patterns mapped to five canonical failure labels. Fast, interpretable,
    and correct for well-known failure signatures.

Stage 2 — ML fallback (TF-IDF + LogisticRegression)
    Trained on a hardcoded synthetic corpus of 80 labelled log snippets so the
    model is immediately usable on first run before real failures accumulate.
    When real failure logs are collected they can be appended to ``CORPUS`` to
    improve accuracy.

Output
------
For each config, writes a ``triage_report.json`` to the config directory and
inserts a row into the ``triage_results`` SQLite table.

Failure mode labels
-------------------
- ``pipeline_stall_deadlock``
- ``cache_coherency_violation``
- ``overflow_in_alu``
- ``reset_sequencing_error``
- ``bus_contention``
- ``unknown``   (ML fallback could not classify with confidence ≥ threshold)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from soc_dse.backend import db
from soc_dse.triage.vcd_parser import FailureWindow, parse_vcd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
CONFIGS_DIR: Path = Path(os.environ.get("SOC_CONFIGS_DIR", _ROOT / "configs"))

# ---------------------------------------------------------------------------
# Failure labels
# ---------------------------------------------------------------------------

LABELS = [
    "pipeline_stall_deadlock",
    "cache_coherency_violation",
    "overflow_in_alu",
    "reset_sequencing_error",
    "bus_contention",
]

# ---------------------------------------------------------------------------
# Stage 1: rule-based patterns
# ---------------------------------------------------------------------------

_RULES: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"stall|deadlock|livelock|backpressure", re.IGNORECASE),
        "pipeline_stall_deadlock",
        "Check pipeline stall/flush logic; verify backpressure handshaking between stages.",
    ),
    (
        re.compile(r"coher|dirty|invalid|MESI|writeback|snoop", re.IGNORECASE),
        "cache_coherency_violation",
        "Inspect cache invalidation protocol; ensure write-back ordering is correct.",
    ),
    (
        re.compile(r"overflow|carry|saturate|wraparound|arithmetic", re.IGNORECASE),
        "overflow_in_alu",
        "Add overflow detection logic to ALU; consider saturating arithmetic.",
    ),
    (
        re.compile(r"reset|rst_n|power.on|por|initialization|init", re.IGNORECASE),
        "reset_sequencing_error",
        "Verify reset de-assertion sequence; ensure all FFs have synchronous reset.",
    ),
    (
        re.compile(r"contention|arbitrat|grant|bus.fault|collision", re.IGNORECASE),
        "bus_contention",
        "Review bus arbitration logic; check for simultaneous drive conflicts.",
    ),
]

_FIX_HINTS: dict[str, str] = {
    rule[1]: rule[2] for rule in _RULES
}
_FIX_HINTS["unknown"] = "No known pattern matched. Inspect VCD waveform manually."


def _rule_classify(text: str) -> Optional[str]:
    """Return the first matching failure label, or None if no rule fires."""
    for pattern, label, _ in _RULES:
        if pattern.search(text):
            return label
    return None


# ---------------------------------------------------------------------------
# Stage 2: synthetic training corpus
# ---------------------------------------------------------------------------

# 80 synthetic log snippets, 16 per class (5 classes × 16 = 80 total)
_CORPUS: list[tuple[str, str]] = [
    # ---- pipeline_stall_deadlock ----
    ("pipeline stall detected at stage 3 backpressure from cache", "pipeline_stall_deadlock"),
    ("livelock condition: stage 2 repeatedly stalling on resource unavailable", "pipeline_stall_deadlock"),
    ("stall signal asserted for 256 consecutive cycles deadlock suspected", "pipeline_stall_deadlock"),
    ("backpressure propagation halted pipeline at fetch stage", "pipeline_stall_deadlock"),
    ("execution unit stall: no forward progress for 128 cycles", "pipeline_stall_deadlock"),
    ("pipeline flush triggered but stall persists downstream", "pipeline_stall_deadlock"),
    ("hazard detection unit raised stall: RAW dependency unresolved", "pipeline_stall_deadlock"),
    ("inter-stage latch frozen: ready signal never asserted", "pipeline_stall_deadlock"),
    ("stall propagation from memory stage blocking decode", "pipeline_stall_deadlock"),
    ("deadlock: producer stall waiting for consumer; consumer stall waiting for producer", "pipeline_stall_deadlock"),
    ("stall count exceeded threshold: possible livelock in pipeline", "pipeline_stall_deadlock"),
    ("writeback stall: no available destination register for 50+ cycles", "pipeline_stall_deadlock"),
    ("instruction queue stall: full buffer backpressure", "pipeline_stall_deadlock"),
    ("pipeline halted: stall from branch misprediction recovery", "pipeline_stall_deadlock"),
    ("stage 5 reporting stall: downstream not consuming results", "pipeline_stall_deadlock"),
    ("critical: deadlock in ring interconnect detected at node 3", "pipeline_stall_deadlock"),

    # ---- cache_coherency_violation ----
    ("cache coherency violation: dirty line evicted without writeback", "cache_coherency_violation"),
    ("MESI protocol error: invalid state transition from E to S", "cache_coherency_violation"),
    ("snoop filter miss: cache line modified by another agent", "cache_coherency_violation"),
    ("coherency error: stale read data from invalidated cache line", "cache_coherency_violation"),
    ("dirty bit inconsistency detected in cache bank 2", "cache_coherency_violation"),
    ("write-back ordering violation: line written back out of order", "cache_coherency_violation"),
    ("cache invalidation missed: coherency protocol failed", "cache_coherency_violation"),
    ("coherent fabric reported: two caches hold modified copy simultaneously", "cache_coherency_violation"),
    ("LLC coherency assertion: tag mismatch between L1 and L2", "cache_coherency_violation"),
    ("snoop response timeout: coherency state unknown", "cache_coherency_violation"),
    ("invalidation broadcast lost: cache line remains in shared state", "cache_coherency_violation"),
    ("write propagation failure: dirty data not visible to other cores", "cache_coherency_violation"),
    ("cache way-conflict causing coherency loss in bank 0", "cache_coherency_violation"),
    ("coher violation: read-modify-write non-atomic across cache boundary", "cache_coherency_violation"),
    ("cache line transitioned to invalid while in use", "cache_coherency_violation"),
    ("data coherency failure: stale value 0xDEADBEEF read after write", "cache_coherency_violation"),

    # ---- overflow_in_alu ----
    ("ALU arithmetic overflow detected on ADD operation", "overflow_in_alu"),
    ("carry bit set unexpectedly: possible wraparound in 32-bit adder", "overflow_in_alu"),
    ("signed overflow: result exceeded representable range", "overflow_in_alu"),
    ("multiply accumulate overflow in MAC unit", "overflow_in_alu"),
    ("saturation logic missing: value wrapped to negative", "overflow_in_alu"),
    ("overflow flag asserted: 0xFFFFFFFF + 1 not saturated", "overflow_in_alu"),
    ("ALU result mismatch: overflow condition not flagged", "overflow_in_alu"),
    ("subtraction underflow: unsigned result wrapped around zero", "overflow_in_alu"),
    ("arithmetic exception: shift amount exceeds operand width", "overflow_in_alu"),
    ("overflow in barrel shifter output", "overflow_in_alu"),
    ("ALU overflow on SUB: negative result in unsigned context", "overflow_in_alu"),
    ("cumulative add overflow after 64 iterations", "overflow_in_alu"),
    ("overflow detected: partial product sum exceeded 32 bits", "overflow_in_alu"),
    ("divide result overflow: quotient does not fit in destination register", "overflow_in_alu"),
    ("integer wraparound on loop counter at cycle 4096", "overflow_in_alu"),
    ("ALU carry chain overflow in ripple adder at bit 31", "overflow_in_alu"),

    # ---- reset_sequencing_error ----
    ("reset de-assertion before PLL lock: undefined state", "reset_sequencing_error"),
    ("rst_n released while clock not stable: flip-flop metastability", "reset_sequencing_error"),
    ("power-on reset sequence violated: SRAM initialized before supply rail stable", "reset_sequencing_error"),
    ("synchronous reset missing on flop Q[3]: retains unknown value after reset", "reset_sequencing_error"),
    ("reset stretcher failed: reset pulse too short for slow domain", "reset_sequencing_error"),
    ("reset released while memory controller still in init phase", "reset_sequencing_error"),
    ("glitch on rst_n during power-on: FSM entered invalid state", "reset_sequencing_error"),
    ("reset sequencing error: module B reset released before module A", "reset_sequencing_error"),
    ("cold reset not propagating to sub-module: register retains stale value", "reset_sequencing_error"),
    ("initialization sequence violated: write before reset complete", "reset_sequencing_error"),
    ("POR circuit failed to assert rst_n for minimum required duration", "reset_sequencing_error"),
    ("async reset crossing clock domain without synchronizer", "reset_sequencing_error"),
    ("reset tree imbalance: child module exits reset 3 cycles before parent", "reset_sequencing_error"),
    ("FSM reset state unreachable: no reset arc defined", "reset_sequencing_error"),
    ("rst_n toggled mid-operation: in-flight transaction corrupted", "reset_sequencing_error"),
    ("warm reset failed to clear pipeline registers", "reset_sequencing_error"),

    # ---- bus_contention ----
    ("bus contention: two masters driving simultaneously on cycle 512", "bus_contention"),
    ("arbitration failure: grant not deasserted before next request", "bus_contention"),
    ("AHB bus collision: multiple agents asserting HADDR concurrently", "bus_contention"),
    ("bus fault: tri-state driver conflict on data lines", "bus_contention"),
    ("AXI contention: write data and read data interleaved incorrectly", "bus_contention"),
    ("bus arbitration timeout: no grant issued within 32 cycles", "bus_contention"),
    ("simultaneous bus access: two peripherals driving same address", "bus_contention"),
    ("bus error: contention between DMA and CPU on write path", "bus_contention"),
    ("protocol violation: slave driving bus before master releases", "bus_contention"),
    ("signal collision on MISO line: two SPI slaves enabled simultaneously", "bus_contention"),
    ("bus hold violation: master releasing bus while slave still driving", "bus_contention"),
    ("arbitration grant collision: two masters received grant simultaneously", "bus_contention"),
    ("bus contention during burst: new master won arbitration mid-burst", "bus_contention"),
    ("I2C bus SDA contention: unexpected low pulled by non-addressed slave", "bus_contention"),
    ("AMBA bus error flag raised: contention on HWDATA", "bus_contention"),
    ("priority inversion in bus arbiter causing contention", "bus_contention"),
]

# ---------------------------------------------------------------------------
# ML pipeline (TF-IDF + LogisticRegression)
# ---------------------------------------------------------------------------

def _build_ml_classifier() -> Pipeline:
    """Build and train the ML classifier on the synthetic corpus."""
    texts = [t for t, _ in _CORPUS]
    labels = [l for _, l in _CORPUS]

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_features=2000,
            sublinear_tf=True,
        )),
        ("lr", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    clf.fit(texts, labels)
    log.debug("ML classifier trained on %d synthetic samples", len(texts))
    return clf


# Singleton — trained once at import time
_ML_CLASSIFIER: Optional[Pipeline] = None


def _get_ml_classifier() -> Pipeline:
    global _ML_CLASSIFIER
    if _ML_CLASSIFIER is None:
        _ML_CLASSIFIER = _build_ml_classifier()
    return _ML_CLASSIFIER


# ---------------------------------------------------------------------------
# Signal extraction from failure windows
# ---------------------------------------------------------------------------

def _extract_affected_signals(windows: list[FailureWindow]) -> list[str]:
    """Return deduplicated list of signal names active in failure windows."""
    seen: set[str] = set()
    signals: list[str] = []
    for w in windows:
        for change in w.context:
            if change.name not in seen:
                seen.add(change.name)
                signals.append(change.name)
    return signals


def _window_to_text(windows: list[FailureWindow]) -> str:
    """Flatten failure window context into a single log-like text string."""
    parts: list[str] = []
    for w in windows:
        parts.append(f"failure at {w.failure_time_ps}ps")
        for change in w.context:
            parts.append(f"{change.name}={change.value}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Two-stage classify
# ---------------------------------------------------------------------------

def classify(
    log_text: str,
    *,
    ml_confidence_threshold: float = 0.50,
) -> tuple[str, str]:
    """
    Classify a failure log string into a failure mode label.

    Parameters
    ----------
    log_text:
        Combined text from simulation log + VCD signal names/values.
    ml_confidence_threshold:
        Minimum posterior probability for the ML classifier to accept a label.
        Below this threshold, returns ``"unknown"``.

    Returns
    -------
    (label, fix_hint)
    """
    # Stage 1: rule-based
    label = _rule_classify(log_text)
    if label:
        return label, _FIX_HINTS[label]

    # Stage 2: ML fallback
    clf = _get_ml_classifier()
    probs = clf.predict_proba([log_text])[0]
    max_prob = probs.max()
    pred_label = clf.classes_[probs.argmax()]

    if max_prob >= ml_confidence_threshold:
        return pred_label, _FIX_HINTS.get(pred_label, _FIX_HINTS["unknown"])

    return "unknown", _FIX_HINTS["unknown"]


# ---------------------------------------------------------------------------
# Per-config triage
# ---------------------------------------------------------------------------

def triage_config(config_id: str) -> Optional[dict]:
    """
    Run full triage for a single config:

    1. Parse VCD for failure windows.
    2. Read sim.log for additional text context.
    3. Classify failure mode.
    4. Write triage_report.json to config dir.
    5. Upsert triage_results in SQLite.

    Returns the triage report dict, or None if no simulation artifacts exist.
    """
    config_dir = CONFIGS_DIR / config_id
    vcd_path = config_dir / "sim.vcd"
    sim_log_path = config_dir / "sim.log"

    if not vcd_path.exists() and not sim_log_path.exists():
        log.warning("[%s] No simulation artifacts found for triage", config_id)
        return None

    # Parse VCD
    windows = parse_vcd(vcd_path) if vcd_path.exists() else []

    # Load sim log text
    sim_log_text = ""
    if sim_log_path.exists():
        try:
            sim_log_text = sim_log_path.read_text(errors="replace")
        except OSError:
            pass

    # Build combined classification text
    vcd_text = _window_to_text(windows)
    combined_text = f"{sim_log_text} {vcd_text}".strip()

    if not combined_text:
        label, fix_hint = "unknown", _FIX_HINTS["unknown"]
        affected_signals: list[str] = []
    else:
        label, fix_hint = classify(combined_text)
        affected_signals = _extract_affected_signals(windows)

    report = {
        "config_id": config_id,
        "failure_mode": label,
        "affected_signals": affected_signals,
        "fix_hint": fix_hint,
        "failure_count": len(windows),
    }

    # Write report
    report_path = config_dir / "triage_report.json"
    with report_path.open("w") as fh:
        json.dump(report, fh, indent=2)

    # Persist to DB
    db.upsert_triage(
        config_id,
        failure_mode=label,
        affected_signals=affected_signals,
        fix_hint=fix_hint,
    )

    log.info(
        "[%s] Triage: %s  signals=%d  hint=%s",
        config_id, label, len(affected_signals), fix_hint[:60],
    )
    return report


def triage_all_failed(config_ids: Optional[list[str]] = None) -> dict[str, Optional[dict]]:
    """
    Run triage on all configs that have assertion violations > 0.

    Returns {config_id → report_dict | None}.
    """
    from soc_dse.backend.db import get_conn

    if config_ids is None:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT config_id FROM simulation_results "
                "WHERE assertion_violations > 0 OR status = 'error'"
            ).fetchall()
        config_ids = [r["config_id"] for r in rows]

    results: dict[str, Optional[dict]] = {}
    for cid in config_ids:
        results[cid] = triage_config(cid)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Triage RTL failure modes from simulation artifacts")
    parser.add_argument("config_ids", nargs="*", help="Config IDs to triage (default: all failed)")
    args = parser.parse_args()

    db.init_db()
    ids = args.config_ids or None
    results = triage_all_failed(ids)
    n_ok = sum(1 for r in results.values() if r is not None)
    print(f"Triaged {n_ok}/{len(results)} configs")


if __name__ == "__main__":
    main()
