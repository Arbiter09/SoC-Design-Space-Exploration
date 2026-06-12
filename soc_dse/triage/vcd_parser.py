"""
Two-pass VCD (Value Change Dump) parser for Verilator output.

Pass 1 — builds a signal symbol table from ``$var`` declarations.
Pass 2 — replays value changes and extracts ±10-cycle context windows
          around each assertion-failure timestamp.

Verilator VCD uses ``$scope``/``$var``/``$upscope`` blocks to declare
signals; these must be parsed before the body can be decoded, which is
why a two-pass approach is required (single-pass regex cannot handle this).

Public API
----------
parse_vcd(vcd_path, clk_period_ps=10_000)
    → list[FailureWindow]
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Number of cycles to capture before and after each failure timestamp
CONTEXT_CYCLES = 10

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SignalInfo:
    """Metadata for one signal extracted from VCD ``$var`` declarations."""
    identifier: str      # VCD symbol code (e.g. "!")
    name: str            # Hierarchical signal name
    width: int           # Bit width
    scope: str           # Parent scope path


@dataclasses.dataclass
class SignalChange:
    """A single value change event."""
    time_ps: int         # Simulation timestamp in picoseconds
    name: str            # Signal name
    value: str           # New value as string ("0", "1", "x", or binary vector)


@dataclasses.dataclass
class FailureWindow:
    """Signal context window around one assertion failure."""
    failure_time_ps: int
    context: list[SignalChange]   # Changes within ±CONTEXT_CYCLES of the failure


# ---------------------------------------------------------------------------
# Pass 1 — Build symbol table from $var declarations
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(
    r"\$var\s+\w+\s+(\d+)\s+(\S+)\s+(\S+)\s+(?:\[[\d:]+\]\s+)?\$end",
    re.IGNORECASE,
)
_SCOPE_RE  = re.compile(r"\$scope\s+\w+\s+(\S+)\s+\$end", re.IGNORECASE)
_UPSCOPE_RE = re.compile(r"\$upscope\s+\$end", re.IGNORECASE)


def _build_symbol_table(vcd_text: str) -> dict[str, SignalInfo]:
    """
    First pass: parse the VCD header to build a {identifier → SignalInfo} map.

    Handles nested ``$scope`` / ``$upscope`` blocks to reconstruct
    hierarchical signal names.
    """
    symbol_table: dict[str, SignalInfo] = {}
    scope_stack: list[str] = []

    # Process header only (up to $dumpvars or first timestamp)
    header_end = vcd_text.find("$dumpvars")
    if header_end == -1:
        header_end = len(vcd_text)
    header = vcd_text[:header_end]

    # Tokenise into directives
    for line in header.splitlines():
        line = line.strip()
        if not line:
            continue

        scope_m = _SCOPE_RE.search(line)
        if scope_m:
            scope_stack.append(scope_m.group(1))
            continue

        if _UPSCOPE_RE.search(line):
            if scope_stack:
                scope_stack.pop()
            continue

        var_m = _VAR_RE.search(line)
        if var_m:
            width = int(var_m.group(1))
            identifier = var_m.group(2)
            sig_name = var_m.group(3)
            scope = ".".join(scope_stack)
            full_name = f"{scope}.{sig_name}" if scope else sig_name
            symbol_table[identifier] = SignalInfo(
                identifier=identifier,
                name=full_name,
                width=width,
                scope=scope,
            )

    log.debug("Symbol table built: %d signals", len(symbol_table))
    return symbol_table


# ---------------------------------------------------------------------------
# Pass 2 — Replay value changes
# ---------------------------------------------------------------------------

_TIME_RE       = re.compile(r"^#(\d+)$")
_SCALAR_RE     = re.compile(r"^([01xzXZ])(\S+)$")
_VECTOR_RE     = re.compile(r"^[br]([01xzXZ]+)\s+(\S+)$", re.IGNORECASE)
_ASSERT_FAIL_RE = re.compile(
    r"(assertion|assert|fail|violation)", re.IGNORECASE
)


def _replay_changes(
    vcd_text: str,
    symbol_table: dict[str, SignalInfo],
    clk_period_ps: int,
) -> tuple[list[SignalChange], list[int]]:
    """
    Second pass: replay all value changes and detect assertion-failure timestamps.

    Returns (all_changes, failure_timestamps_ps).

    A failure is detected when a signal matching ``ASSERT_FAIL_RE`` transitions
    to 1 (logic high), which is the Verilator convention for assertion outputs.
    """
    all_changes: list[SignalChange] = []
    failure_times: list[int] = []
    current_time: int = 0
    in_dumpvars = False

    for line in vcd_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("$dumpvars"):
            in_dumpvars = True
            continue
        if line.startswith("$end") and in_dumpvars:
            in_dumpvars = False
            continue

        time_m = _TIME_RE.match(line)
        if time_m:
            current_time = int(time_m.group(1))
            continue

        # Scalar value change: 0!, 1!, x!, z!
        scalar_m = _SCALAR_RE.match(line)
        if scalar_m:
            value = scalar_m.group(1)
            ident = scalar_m.group(2)
            info = symbol_table.get(ident)
            if info:
                change = SignalChange(
                    time_ps=current_time,
                    name=info.name,
                    value=value,
                )
                all_changes.append(change)
                # Detect assertion failures: signal named *assert*/*fail* goes high
                if value == "1" and _ASSERT_FAIL_RE.search(info.name):
                    failure_times.append(current_time)
            continue

        # Vector value change: b01010 !, r1.5 !
        vec_m = _VECTOR_RE.match(line)
        if vec_m:
            value = vec_m.group(1)
            ident = vec_m.group(2)
            info = symbol_table.get(ident)
            if info:
                all_changes.append(SignalChange(
                    time_ps=current_time,
                    name=info.name,
                    value=value,
                ))

    return all_changes, failure_times


# ---------------------------------------------------------------------------
# Context window extraction
# ---------------------------------------------------------------------------

def _extract_windows(
    all_changes: list[SignalChange],
    failure_times: list[int],
    clk_period_ps: int,
) -> list[FailureWindow]:
    """
    For each failure timestamp, extract all signal changes within
    ±CONTEXT_CYCLES clock periods.
    """
    window_ps = CONTEXT_CYCLES * clk_period_ps
    windows: list[FailureWindow] = []

    for ft in failure_times:
        lo = ft - window_ps
        hi = ft + window_ps
        context = [c for c in all_changes if lo <= c.time_ps <= hi]
        windows.append(FailureWindow(failure_time_ps=ft, context=context))

    return windows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_vcd(
    vcd_path: Path,
    *,
    clk_period_ps: int = 10_000,  # 100 MHz default
) -> list[FailureWindow]:
    """
    Parse a Verilator VCD file and return failure context windows.

    Parameters
    ----------
    vcd_path:
        Path to the ``.vcd`` file produced by the Verilator testbench.
    clk_period_ps:
        Clock period in picoseconds (default: 10 000 ps = 100 MHz).

    Returns
    -------
    list[FailureWindow]
        One entry per assertion failure found in the VCD.
        Empty list if no failures were detected.
    """
    if not vcd_path.exists():
        log.warning("VCD file not found: %s", vcd_path)
        return []

    try:
        vcd_text = vcd_path.read_text(errors="replace")
    except OSError as exc:
        log.error("Cannot read VCD file %s: %s", vcd_path, exc)
        return []

    symbol_table = _build_symbol_table(vcd_text)
    if not symbol_table:
        log.warning("No signals found in VCD header: %s", vcd_path)
        return []

    all_changes, failure_times = _replay_changes(vcd_text, symbol_table, clk_period_ps)
    log.debug(
        "VCD replay: %d changes, %d assertion failures", len(all_changes), len(failure_times)
    )

    if not failure_times:
        return []

    return _extract_windows(all_changes, failure_times, clk_period_ps)
