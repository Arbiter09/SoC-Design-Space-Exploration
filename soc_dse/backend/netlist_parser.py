"""
Yosys JSON netlist → NetworkX graph with node features.

Parses the ``netlist.json`` produced by Yosys ``write_json`` and builds a
directed graph suitable for PyTorch Geometric ingestion:

Node features (per cell)
------------------------
- ``cell_type_onehot`` : one-hot vector over the top 16 Yosys cell types (len=16)
- ``fanin``            : number of input bits connected to this cell
- ``fanout``           : number of output bits driven by this cell
- ``logic_depth``      : BFS depth from primary inputs (longest path)

Graph-level labels
------------------
- ``area``             : from synthesis_results DB row
- ``delay``            : from synthesis_results DB row
- ``power_estimate``   : from synthesis_results DB row

The serialised graph is written to ``GRAPHS_DIR/<config_id>.json``.

Environment variables
---------------------
SOC_CONFIGS_DIR   Root directory for generated configs (default: soc_dse/configs/)
SOC_GRAPHS_DIR    Output directory for graph JSON files (default: soc_dse/graphs/)
SOC_DB_PATH       SQLite database path (default: soc_dse/dse.db)
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from soc_dse.backend import db

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
CONFIGS_DIR: Path = Path(os.environ.get("SOC_CONFIGS_DIR", _ROOT / "configs"))
GRAPHS_DIR: Path = Path(os.environ.get("SOC_GRAPHS_DIR", _ROOT / "graphs"))

# ---------------------------------------------------------------------------
# Cell-type vocabulary — top 16 Yosys internal cell types
# ---------------------------------------------------------------------------

CELL_TYPES: list[str] = [
    "$and", "$or", "$xor", "$not", "$mux",
    "$dff", "$dffe", "$add", "$sub", "$mul",
    "$eq", "$lt", "$gt", "$reduce_and", "$reduce_or", "$pmux",
]
CELL_TYPE_INDEX: dict[str, int] = {ct: i for i, ct in enumerate(CELL_TYPES)}
NUM_CELL_TYPES = len(CELL_TYPES)  # 16


def _cell_type_onehot(cell_type: str) -> list[int]:
    """Return a length-16 one-hot vector for *cell_type* (unknown → all zeros)."""
    vec = [0] * NUM_CELL_TYPES
    idx = CELL_TYPE_INDEX.get(cell_type.lower())
    if idx is not None:
        vec[idx] = 1
    return vec


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph(netlist: dict[str, Any], top_name: str) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from a Yosys JSON netlist module.

    Nodes are cell names; edges represent wire connections between cells.
    Each node carries raw attributes (cell_type, connections) used later
    to compute fanin/fanout and depth features.
    """
    top = netlist["modules"][top_name]
    cells: dict[str, dict] = top.get("cells", {})
    netnames: dict[str, dict] = top.get("netnames", {})

    G = nx.DiGraph()

    # Add one node per cell
    for cell_name, cell_info in cells.items():
        G.add_node(cell_name, cell_type=cell_info.get("type", "unknown"))

    # Build a signal-bit → cell mapping for edge construction
    # Each port direction tells us if the cell drives (output) or reads (input) a bit
    bit_driven_by: dict[int, str] = {}   # bit_id → cell_name that drives it
    bit_read_by: dict[int, list[str]] = {}  # bit_id → [cell_names] that read it

    for cell_name, cell_info in cells.items():
        port_directions: dict[str, str] = cell_info.get("port_directions", {})
        connections: dict[str, list] = cell_info.get("connections", {})

        for port_name, bits in connections.items():
            direction = port_directions.get(port_name, "input")
            for bit in bits:
                if not isinstance(bit, int):
                    continue  # skip "x" / "z" constants
                if direction == "output":
                    bit_driven_by[bit] = cell_name
                else:
                    bit_read_by.setdefault(bit, []).append(cell_name)

    # Add edges: driver → reader
    for bit, driver in bit_driven_by.items():
        for reader in bit_read_by.get(bit, []):
            if driver != reader:
                G.add_edge(driver, reader)

    return G


def _compute_node_features(
    G: nx.DiGraph,
    cells: dict[str, dict],
) -> dict[str, dict[str, Any]]:
    """
    Compute numeric node features for all cells in the graph.

    Returns a dict mapping cell_name → feature dict with keys:
    ``cell_type_onehot``, ``fanin``, ``fanout``, ``logic_depth``.
    """
    # fanin / fanout from graph topology
    fanin_map: dict[str, int] = {n: G.in_degree(n) for n in G.nodes}
    fanout_map: dict[str, int] = {n: G.out_degree(n) for n in G.nodes}

    # Logic depth: BFS from source nodes (no predecessors) using longest-path
    # approximation via topological sort
    depth: dict[str, int] = {n: 0 for n in G.nodes}
    try:
        for node in nx.topological_sort(G):
            for successor in G.successors(node):
                depth[successor] = max(depth[successor], depth[node] + 1)
    except nx.NetworkXUnfeasible:
        # Cycle present (unlikely in synthesised netlist but guard anyway)
        log.warning("Cycle detected in netlist graph — depth set to 0 for all nodes")

    features: dict[str, dict[str, Any]] = {}
    for cell_name in G.nodes:
        cell_type = G.nodes[cell_name].get("cell_type", "unknown")
        features[cell_name] = {
            "cell_type_onehot": _cell_type_onehot(cell_type),
            "fanin": fanin_map[cell_name],
            "fanout": fanout_map[cell_name],
            "logic_depth": depth[cell_name],
        }

    return features


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _graph_to_dict(
    G: nx.DiGraph,
    features: dict[str, dict[str, Any]],
    config_id: str,
    ppa: dict[str, float],
) -> dict[str, Any]:
    """
    Serialise the graph + features to a plain dict for JSON export.

    Format
    ------
    {
      "config_id": "...",
      "nodes": [{"id": "...", "features": {...}}, ...],
      "edges": [{"src": "...", "dst": "..."}, ...],
      "labels": {"area": ..., "delay": ..., "power_estimate": ...}
    }
    """
    node_index = {name: i for i, name in enumerate(G.nodes)}
    nodes = [
        {"id": name, "idx": node_index[name], "features": features[name]}
        for name in G.nodes
    ]
    edges = [
        {"src": node_index[u], "dst": node_index[v]}
        for u, v in G.edges
    ]
    return {
        "config_id": config_id,
        "nodes": nodes,
        "edges": edges,
        "labels": ppa,
        "num_node_features": NUM_CELL_TYPES + 3,  # onehot(16) + fanin + fanout + depth
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_netlist(config_id: str) -> Path | None:
    """
    Parse the Yosys JSON netlist for *config_id* and write the graph to
    ``GRAPHS_DIR/<config_id>.json``.

    Returns the output path on success, None on failure.
    """
    netlist_path = CONFIGS_DIR / config_id / "netlist.json"
    if not netlist_path.exists():
        log.warning("[%s] netlist.json not found — run synthesis first", config_id)
        return None

    # Load PPA labels from DB
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT area, delay, power_estimate FROM synthesis_results "
            "WHERE config_id = ? AND status = 'ok'",
            (config_id,),
        ).fetchone()

    if row is None:
        log.warning("[%s] No successful synthesis result in DB — skipping graph", config_id)
        return None

    ppa = {
        "area": row["area"],
        "delay": row["delay"],
        "power_estimate": row["power_estimate"],
    }

    try:
        with netlist_path.open() as fh:
            netlist: dict = json.load(fh)

        modules = netlist.get("modules", {})
        top_name = "soc_top" if "soc_top" in modules else next(iter(modules), None)
        if top_name is None:
            log.error("[%s] No modules in netlist JSON", config_id)
            return None

        G = _build_graph(netlist, top_name)
        cells = netlist["modules"][top_name].get("cells", {})
        features = _compute_node_features(G, cells)
        graph_dict = _graph_to_dict(G, features, config_id, ppa)

        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = GRAPHS_DIR / f"{config_id}.json"
        with out_path.open("w") as fh:
            json.dump(graph_dict, fh)

        log.info(
            "[%s] Graph saved: %d nodes, %d edges → %s",
            config_id, len(G.nodes), len(G.edges), out_path,
        )
        return out_path

    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] Failed to parse netlist: %s", config_id, exc)
        return None


def parse_all_netlists(config_ids: list[str] | None = None) -> dict[str, bool]:
    """
    Parse netlists for all configs (or the given subset).

    Returns {config_id: success}.
    """
    if config_ids is None:
        config_ids = db.get_all_config_ids()

    results: dict[str, bool] = {}
    for cid in config_ids:
        results[cid] = parse_netlist(cid) is not None
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Parse Yosys netlists → graph JSON files")
    parser.add_argument(
        "config_ids",
        nargs="*",
        help="Specific config IDs to parse (default: all in DB)",
    )
    args = parser.parse_args()

    db.init_db()
    ids = args.config_ids or None
    results = parse_all_netlists(ids)
    n_ok = sum(results.values())
    print(f"Parsed {n_ok}/{len(results)} graphs successfully")


if __name__ == "__main__":
    main()
