"""
GNN inference with MC Dropout confidence intervals.

Given a path to a netlist graph JSON (already parsed by ``netlist_parser.py``),
loads the best checkpoint, runs N stochastic forward passes (MC Dropout),
and returns per-target mean predictions with ±1σ confidence intervals.

Also validates that the input graph's node feature dimension matches the
checkpoint — mismatches silently produce wrong results otherwise.

Usage
-----
    python -m soc_dse.model.predict <graph_json_path> [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data

from soc_dse.model.dataset import (
    GRAPHS_DIR,
    NUM_NODE_FEATURES,
    PPA_KEYS,
    NetlistDataset,
)
from soc_dse.model.gnn import PPAGNN
from soc_dse.model.train import BEST_CHECKPOINT

log = logging.getLogger(__name__)

_DEFAULT_MC_SAMPLES = 30


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path = BEST_CHECKPOINT) -> tuple[PPAGNN, dict]:
    """
    Load the best checkpoint and return (model, checkpoint_dict).

    Raises FileNotFoundError if the checkpoint does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Run `python -m soc_dse.model.train` first."
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = PPAGNN(
        num_node_features=ckpt["num_node_features"],
        hidden_dim=ckpt["hidden_dim"],
        dropout_p=ckpt["dropout_p"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Graph loading (single file)
# ---------------------------------------------------------------------------

def _load_graph_file(graph_json: Path) -> Data:
    """Load a single graph JSON file into a ``torch_geometric.data.Data`` object."""
    with graph_json.open() as fh:
        g = json.load(fh)

    nodes = g["nodes"]
    edges = g["edges"]

    x_rows: list[list[float]] = []
    for node in nodes:
        feat = node["features"]
        onehot: list[int] = feat["cell_type_onehot"]
        row = onehot + [
            float(feat["fanin"]),
            float(feat["fanout"]),
            float(feat["logic_depth"]),
        ]
        x_rows.append(row)

    x = torch.tensor(x_rows, dtype=torch.float)

    if edges:
        src = [e["src"] for e in edges]
        dst = [e["dst"] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)  # single graph → batch = 0
    data.config_id = g.get("config_id", graph_json.stem)
    return data


# ---------------------------------------------------------------------------
# Feature dimension validation
# ---------------------------------------------------------------------------

def _validate_feature_dim(data: Data, model: PPAGNN, checkpoint_path: Path) -> None:
    """
    Warn if the input graph's feature dimension does not match the checkpoint.

    A mismatch causes the model to produce garbage predictions without
    raising an error, so we surface it explicitly.
    """
    input_dim = data.x.shape[1]
    ckpt_dim = model.conv1.in_channels  # SAGEConv stores in_channels

    if input_dim != ckpt_dim:
        warnings.warn(
            f"Input graph has {input_dim} node features, but the checkpoint "
            f"at '{checkpoint_path}' was trained with {ckpt_dim}. "
            "Predictions will be incorrect. Re-train the model or re-parse "
            "the netlist with the same feature schema.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_ci(
    graph_json: Path,
    *,
    checkpoint_path: Path = BEST_CHECKPOINT,
    graphs_dir: Optional[Path] = None,
    n_samples: int = _DEFAULT_MC_SAMPLES,
) -> dict[str, dict[str, float]]:
    """
    Predict PPA for a single netlist graph using MC Dropout.

    Parameters
    ----------
    graph_json:
        Path to the ``<config_id>.json`` graph file.
    checkpoint_path:
        Path to the model checkpoint (default: ``model/checkpoints/best.pt``).
    graphs_dir:
        Directory used to find ``norm_stats.json`` (default: ``soc_dse/graphs/``).
    n_samples:
        Number of stochastic forward passes (default: 30).

    Returns
    -------
    dict
        Mapping PPA key → {"mean": float, "std": float, "mean_denorm": float}
        ``mean`` / ``std`` are in normalised space; ``mean_denorm`` is the
        original-scale prediction.
    """
    model, ckpt = load_model(checkpoint_path)
    data = _load_graph_file(graph_json)
    _validate_feature_dim(data, model, checkpoint_path)

    # Enable MC Dropout (stochastic even in eval mode)
    model.enable_mc_dropout()

    samples: list[torch.Tensor] = []
    for _ in range(n_samples):
        out = model(data.x, data.edge_index, data.batch)  # [1, 3]
        samples.append(out.squeeze(0))  # [3]

    stacked = torch.stack(samples, dim=0)  # [n_samples, 3]
    means = stacked.mean(dim=0)   # [3]
    stds = stacked.std(dim=0)     # [3]

    # Denormalise using cached stats
    gdir = graphs_dir or GRAPHS_DIR
    norm_stats = NetlistDataset.load_norm_stats(gdir)
    means_denorm = torch.tensor(
        [means[i].item() * norm_stats[k]["std"] + norm_stats[k]["mean"]
         for i, k in enumerate(PPA_KEYS)]
    )

    results: dict[str, dict[str, float]] = {}
    for i, key in enumerate(PPA_KEYS):
        results[key] = {
            "mean_norm": round(means[i].item(), 6),
            "std_norm": round(stds[i].item(), 6),
            "mean": round(means_denorm[i].item(), 4),
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Predict PPA for a netlist graph (MC Dropout CI)")
    parser.add_argument("graph_json", type=Path, help="Path to <config_id>.json graph file")
    parser.add_argument(
        "--checkpoint", type=Path, default=BEST_CHECKPOINT, help="Checkpoint path"
    )
    parser.add_argument(
        "--graphs-dir", type=Path, default=GRAPHS_DIR, help="Directory with norm_stats.json"
    )
    parser.add_argument(
        "--samples", type=int, default=_DEFAULT_MC_SAMPLES, help="MC Dropout forward passes"
    )
    args = parser.parse_args()

    results = predict_with_ci(
        args.graph_json,
        checkpoint_path=args.checkpoint,
        graphs_dir=args.graphs_dir,
        n_samples=args.samples,
    )

    print(f"\nPPA predictions for: {args.graph_json.name}")
    print(f"  (MC Dropout, {args.samples} samples)\n")
    for key, vals in results.items():
        print(
            f"  {key:20s}  {vals['mean']:>10.4f}  ±{vals['std_norm']:.4f} (norm)"
        )


if __name__ == "__main__":
    main()
