"""
PyTorch Geometric Dataset for SoC netlist graphs.

Loads pre-built graph JSON files from ``GRAPHS_DIR`` and exposes them as
``torch_geometric.data.Data`` objects with:

- ``x``       : node feature matrix  [N, NUM_NODE_FEATURES]
- ``edge_index`` : COO edge list     [2, E]
- ``y``       : PPA label vector     [3]  (area, delay, power_estimate)
- ``config_id``: string identifier

PPA labels are normalised (z-score) using statistics computed over the full
dataset. Normalisation parameters are cached in a ``norm_stats.json`` file
inside ``GRAPHS_DIR`` so they can be reused at inference time.

Environment variables
---------------------
SOC_GRAPHS_DIR   Directory containing graph JSON files (default: soc_dse/graphs/)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data, InMemoryDataset

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
GRAPHS_DIR: Path = Path(os.environ.get("SOC_GRAPHS_DIR", _ROOT / "graphs"))

# Node feature dimension: 16 (onehot) + 3 (fanin, fanout, depth) = 19
NUM_NODE_FEATURES = 19
# PPA label order
PPA_KEYS = ["area", "delay", "power_estimate"]


class NetlistDataset(InMemoryDataset):
    """
    In-memory PyTorch Geometric dataset of synthesised SoC netlists.

    Parameters
    ----------
    graphs_dir:
        Directory containing ``<config_id>.json`` graph files.
    transform:
        Optional PyG transform applied at access time.
    """

    def __init__(
        self,
        graphs_dir: Optional[Path] = None,
        transform=None,
    ) -> None:
        self.graphs_dir: Path = graphs_dir or GRAPHS_DIR
        super().__init__(root=str(self.graphs_dir), transform=transform)

        graph_files = sorted(self.graphs_dir.glob("*.json"))
        # Exclude the cached norm_stats file
        graph_files = [f for f in graph_files if f.name != "norm_stats.json"]

        if not graph_files:
            raise RuntimeError(
                f"No graph JSON files found in '{self.graphs_dir}'. "
                "Run `python -m soc_dse.backend.run_pipeline` followed by "
                "`python -m soc_dse.backend.netlist_parser` first to generate graphs."
            )

        data_list = [self._load_graph(f) for f in graph_files]
        data_list = [d for d in data_list if d is not None]

        if not data_list:
            raise RuntimeError(
                "All graph files failed to load. "
                "Check that synthesis ran successfully and netlist.json files exist."
            )

        # Normalise PPA labels and cache stats
        data_list, self.norm_stats = self._normalise_labels(data_list)
        self._save_norm_stats()

        self.data, self.slices = self.collate(data_list)
        log.info("Dataset loaded: %d graphs, %d node features", len(data_list), NUM_NODE_FEATURES)

    # ------------------------------------------------------------------
    # Required InMemoryDataset overrides
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return []

    def download(self) -> None:
        pass

    def process(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_graph(self, path: Path) -> Optional[Data]:
        """Parse a single graph JSON file into a ``torch_geometric.data.Data``."""
        try:
            with path.open() as fh:
                g = json.load(fh)

            nodes = g["nodes"]
            edges = g["edges"]
            labels = g["labels"]

            # Node feature matrix [N, 19]
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

            if not x_rows:
                log.warning("Empty graph: %s — skipping", path.name)
                return None

            x = torch.tensor(x_rows, dtype=torch.float)

            # Edge index [2, E]
            if edges:
                src = [e["src"] for e in edges]
                dst = [e["dst"] for e in edges]
                edge_index = torch.tensor([src, dst], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # PPA label [3]
            y = torch.tensor(
                [labels.get(k, 0.0) for k in PPA_KEYS],
                dtype=torch.float,
            )

            data = Data(x=x, edge_index=edge_index, y=y)
            data.config_id = g.get("config_id", path.stem)
            return data

        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load graph %s: %s", path.name, exc)
            return None

    def _normalise_labels(
        self, data_list: list[Data]
    ) -> tuple[list[Data], dict[str, dict[str, float]]]:
        """
        Z-score normalise PPA targets across the dataset.

        Returns the modified data list and a stats dict with per-target
        ``mean`` and ``std`` for later denormalisation.
        """
        ys = torch.stack([d.y for d in data_list])  # [N, 3]
        means = ys.mean(dim=0)
        stds = ys.std(dim=0).clamp(min=1e-8)

        for d in data_list:
            d.y = (d.y - means) / stds

        stats: dict[str, dict[str, float]] = {
            key: {"mean": means[i].item(), "std": stds[i].item()}
            for i, key in enumerate(PPA_KEYS)
        }
        return data_list, stats

    def _save_norm_stats(self) -> None:
        """Persist normalisation statistics for inference-time denormalisation."""
        stats_path = self.graphs_dir / "norm_stats.json"
        with stats_path.open("w") as fh:
            json.dump(self.norm_stats, fh, indent=2)
        log.debug("Normalisation stats saved to %s", stats_path)

    def denormalise(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised PPA predictions back to original scale."""
        means = torch.tensor([self.norm_stats[k]["mean"] for k in PPA_KEYS])
        stds = torch.tensor([self.norm_stats[k]["std"] for k in PPA_KEYS])
        return y_norm * stds + means

    @staticmethod
    def load_norm_stats(graphs_dir: Optional[Path] = None) -> dict[str, dict[str, float]]:
        """Load cached normalisation statistics from disk."""
        gdir = graphs_dir or GRAPHS_DIR
        stats_path = gdir / "norm_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"norm_stats.json not found in {gdir}. "
                "The dataset must be instantiated at least once before inference."
            )
        with stats_path.open() as fh:
            return json.load(fh)
