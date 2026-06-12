"""
Design Space Exploration visualisation suite.

Generates four plots after a full pipeline run:

1. Pareto frontier       — area vs delay, coloured by pipeline_stages
2. PPA scatter matrix    — pairwise scatter of area / delay / power across configs
3. GNN prediction vs actual — scatter with R² annotation per PPA target
4. Failure rate heatmap  — failure rate (%) by config parameter combinations

Outputs are saved as both PNG (static) and HTML (interactive Plotly) to
``SOC_VIZ_DIR`` (default: ``soc_dse/viz/output/``).

Usage
-----
    python -m soc_dse.viz.plot [--out-dir PATH]
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from soc_dse.backend import db

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
VIZ_DIR: Path = Path(os.environ.get("SOC_VIZ_DIR", _ROOT / "viz" / "output"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results_df() -> pd.DataFrame:
    """
    Join configs + synthesis_results + simulation_results into a single DataFrame.
    """
    with db.get_conn() as conn:
        query = """
            SELECT
                c.config_id,
                c.params_json,
                s.cell_count,
                s.wire_count,
                s.area,
                s.delay,
                s.power_estimate,
                sim.assertion_violations,
                sim.cycle_count,
                sim.toggle_coverage
            FROM configs c
            LEFT JOIN synthesis_results s   ON c.config_id = s.config_id  AND s.status = 'ok'
            LEFT JOIN simulation_results sim ON c.config_id = sim.config_id
        """
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return df

    # Expand params_json into columns
    params_expanded = df["params_json"].apply(json.loads).apply(pd.Series)
    df = pd.concat([df.drop(columns=["params_json"]), params_expanded], axis=1)
    return df


# ---------------------------------------------------------------------------
# 1. Pareto frontier
# ---------------------------------------------------------------------------

def _is_pareto(costs: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of Pareto-optimal points (minimise both objectives).
    """
    n = costs.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                dominated[i] = True
                break
    return ~dominated


def plot_pareto_frontier(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Pareto frontier: area vs delay, coloured by pipeline_stages."""
    sub = df.dropna(subset=["area", "delay"])
    if sub.empty:
        log.warning("No synthesis data for Pareto plot")
        return []

    stages_col = "pipeline_stages" if "pipeline_stages" in sub.columns else None
    costs = sub[["area", "delay"]].values.astype(float)
    pareto_mask = _is_pareto(costs)

    # ---- Matplotlib static PNG ----
    fig, ax = plt.subplots(figsize=(9, 6))
    stages_vals = sub[stages_col].unique() if stages_col else [None]
    cmap = plt.get_cmap("tab10")
    color_map: dict = {}

    for idx, sv in enumerate(sorted(stages_vals)):
        mask = (sub[stages_col] == sv) if stages_col else np.ones(len(sub), dtype=bool)
        color = cmap(idx % 10)
        color_map[sv] = color
        ax.scatter(
            sub.loc[mask, "area"],
            sub.loc[mask, "delay"],
            color=color, alpha=0.55, s=40, label=f"stages={sv}",
        )

    # Highlight Pareto front
    pareto_pts = sub[pareto_mask].sort_values("area")
    ax.plot(
        pareto_pts["area"], pareto_pts["delay"],
        "k--", linewidth=1.5, label="Pareto front",
    )
    ax.scatter(
        pareto_pts["area"], pareto_pts["delay"],
        color="black", zorder=5, s=60,
    )

    ax.set_xlabel("Area (cell units)", fontsize=12)
    ax.set_ylabel("Delay (avg fan-in depth)", fontsize=12)
    ax.set_title("Pareto Frontier: Area vs Delay", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()

    png_path = out_dir / "pareto_frontier.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # ---- Plotly interactive HTML ----
    color_seq = px.colors.qualitative.Plotly
    pfig = px.scatter(
        sub,
        x="area", y="delay",
        color=stages_col if stages_col else None,
        hover_data=["config_id", "area", "delay", "power_estimate"],
        title="Pareto Frontier: Area vs Delay",
        labels={"area": "Area (cell units)", "delay": "Delay (fan-in depth)"},
        color_discrete_sequence=color_seq,
    )
    # Add Pareto front line
    pfig.add_trace(go.Scatter(
        x=pareto_pts["area"].tolist(), y=pareto_pts["delay"].tolist(),
        mode="lines+markers",
        line=dict(color="black", dash="dash", width=2),
        marker=dict(size=8, color="black"),
        name="Pareto front",
    ))
    html_path = out_dir / "pareto_frontier.html"
    pfig.write_html(str(html_path))

    log.info("Pareto plot saved: %s, %s", png_path, html_path)
    return [png_path, html_path]


# ---------------------------------------------------------------------------
# 2. PPA scatter matrix
# ---------------------------------------------------------------------------

def plot_ppa_scatter_matrix(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Pairwise scatter matrix for area / delay / power_estimate."""
    sub = df.dropna(subset=["area", "delay", "power_estimate"])
    if sub.empty:
        log.warning("No data for PPA scatter matrix")
        return []

    dimensions = [
        dict(label="Area", values=sub["area"]),
        dict(label="Delay", values=sub["delay"]),
        dict(label="Power", values=sub["power_estimate"]),
    ]

    pfig = go.Figure(data=go.Splom(
        dimensions=dimensions,
        showupperhalf=False,
        marker=dict(size=5, opacity=0.6, colorscale="Viridis",
                    color=sub.get("pipeline_stages", sub["area"])),
        text=sub["config_id"] if "config_id" in sub.columns else None,
    ))
    pfig.update_layout(title="PPA Scatter Matrix", height=700, width=700)

    html_path = out_dir / "ppa_scatter_matrix.html"
    pfig.write_html(str(html_path))

    # Static PNG via matplotlib
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    ppa_cols = ["area", "delay", "power_estimate"]
    for i, yi in enumerate(ppa_cols):
        for j, xj in enumerate(ppa_cols):
            ax = axes[i][j]
            if i == j:
                ax.hist(sub[yi].dropna(), bins=15, color="steelblue", edgecolor="white")
                ax.set_xlabel(yi)
            elif i > j:
                ax.scatter(sub[xj], sub[yi], s=10, alpha=0.5, color="steelblue")
                ax.set_xlabel(xj)
                ax.set_ylabel(yi)
            else:
                ax.set_visible(False)
    plt.suptitle("PPA Scatter Matrix", fontsize=13)
    plt.tight_layout()
    png_path = out_dir / "ppa_scatter_matrix.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    log.info("PPA scatter matrix saved: %s, %s", png_path, html_path)
    return [png_path, html_path]


# ---------------------------------------------------------------------------
# 3. GNN prediction vs actual
# ---------------------------------------------------------------------------

def plot_gnn_vs_actual(
    predictions: list[dict[str, Any]],
    actuals: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    """
    Scatter plot of GNN-predicted vs actual synthesis PPA.

    Parameters
    ----------
    predictions:
        List of dicts {config_id, area, delay, power_estimate} — predicted values.
    actuals:
        List of dicts {config_id, area, delay, power_estimate} — ground-truth values.
    """
    pred_df = pd.DataFrame(predictions).set_index("config_id")
    act_df = pd.DataFrame(actuals).set_index("config_id")

    common = pred_df.index.intersection(act_df.index)
    if common.empty:
        log.warning("No overlapping configs for GNN vs actual plot")
        return []

    pred_df = pred_df.loc[common]
    act_df = act_df.loc[common]

    ppa_cols = ["area", "delay", "power_estimate"]
    saved: list[Path] = []

    # Matplotlib: one subplot per PPA target
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ppa_cols):
        x = act_df[col].values.astype(float)
        y = pred_df[col].values.astype(float)
        # R²
        ss_res = ((x - y) ** 2).sum()
        ss_tot = ((x - x.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)

        ax.scatter(x, y, alpha=0.65, s=30, color="steelblue")
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="y=x")
        ax.set_xlabel(f"Actual {col}", fontsize=10)
        ax.set_ylabel(f"Predicted {col}", fontsize=10)
        ax.set_title(f"{col}\nR²={r2:.3f}", fontsize=11)
        ax.legend(fontsize=8)

    plt.suptitle("GNN Prediction vs Actual Synthesis", fontsize=13)
    plt.tight_layout()
    png_path = out_dir / "gnn_vs_actual.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    saved.append(png_path)

    # Plotly: faceted scatter
    rows_list: list[dict] = []
    for col in ppa_cols:
        for cid in common:
            rows_list.append({
                "config_id": cid,
                "target": col,
                "actual": act_df.loc[cid, col],
                "predicted": pred_df.loc[cid, col],
            })
    plot_df = pd.DataFrame(rows_list)
    pfig = px.scatter(
        plot_df, x="actual", y="predicted", facet_col="target",
        hover_data=["config_id"],
        title="GNN Prediction vs Actual",
        trendline="ols",
    )
    html_path = out_dir / "gnn_vs_actual.html"
    pfig.write_html(str(html_path))
    saved.append(html_path)

    log.info("GNN vs actual plots saved: %s, %s", png_path, html_path)
    return saved


# ---------------------------------------------------------------------------
# 4. Failure rate heatmap
# ---------------------------------------------------------------------------

def plot_failure_heatmap(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """
    Heatmap of assertion failure rates grouped by two config parameters.

    Uses pipeline_stages × cache_size_kb as axes if available.
    """
    if "assertion_violations" not in df.columns:
        log.warning("No simulation data for failure heatmap")
        return []

    row_param = "pipeline_stages" if "pipeline_stages" in df.columns else None
    col_param = "cache_size_kb" if "cache_size_kb" in df.columns else None

    if row_param is None or col_param is None:
        log.warning("Required parameter columns missing for heatmap")
        return []

    df = df.copy()
    df["failed"] = (df["assertion_violations"].fillna(0) > 0).astype(int)
    pivot = df.groupby([row_param, col_param])["failed"].mean().unstack() * 100

    if pivot.empty:
        log.warning("No data for heatmap pivot table")
        return []

    saved: list[Path] = []

    # Matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v}KB" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v}-stage" for v in pivot.index])
    ax.set_xlabel("Cache Size (KB)", fontsize=11)
    ax.set_ylabel("Pipeline Stages", fontsize=11)
    ax.set_title("Assertion Failure Rate (%) by Config Parameters", fontsize=12)
    plt.colorbar(im, ax=ax, label="Failure rate (%)")
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.0f}%", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    png_path = out_dir / "failure_heatmap.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    saved.append(png_path)

    # Plotly
    pfig = px.imshow(
        pivot,
        labels=dict(x="Cache Size (KB)", y="Pipeline Stages", color="Failure rate (%)"),
        title="Assertion Failure Rate (%) by Config Parameters",
        color_continuous_scale="YlOrRd",
        zmin=0, zmax=100,
        text_auto=".0f",
    )
    html_path = out_dir / "failure_heatmap.html"
    pfig.write_html(str(html_path))
    saved.append(html_path)

    log.info("Failure heatmap saved: %s, %s", png_path, html_path)
    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_plots(
    out_dir: Optional[Path] = None,
    predictions: Optional[list[dict[str, Any]]] = None,
    actuals: Optional[list[dict[str, Any]]] = None,
) -> dict[str, list[Path]]:
    """
    Generate all four visualisations and save to *out_dir*.

    Parameters
    ----------
    out_dir:
        Output directory (default: ``soc_dse/viz/output/``).
    predictions / actuals:
        GNN prediction and actual results for the GNN vs actual plot.
        If omitted, that plot is skipped.

    Returns
    -------
    dict mapping plot name → list of saved file paths
    """
    out = out_dir or VIZ_DIR
    out.mkdir(parents=True, exist_ok=True)

    db.init_db()
    df = _load_results_df()

    saved: dict[str, list[Path]] = {}

    if df.empty:
        log.warning("No results in database — run the pipeline first")
        return saved

    saved["pareto"] = plot_pareto_frontier(df, out)
    saved["scatter_matrix"] = plot_ppa_scatter_matrix(df, out)

    if predictions and actuals:
        saved["gnn_vs_actual"] = plot_gnn_vs_actual(predictions, actuals, out)

    saved["failure_heatmap"] = plot_failure_heatmap(df, out)

    log.info("All plots saved to %s", out)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate DSE visualisation plots")
    parser.add_argument(
        "--out-dir", type=Path, default=VIZ_DIR, help="Output directory for plots"
    )
    parser.add_argument(
        "--gnn-predictions", type=Path, default=None,
        help="JSON file with GNN predictions [{config_id, area, delay, power_estimate}, ...]",
    )
    parser.add_argument(
        "--gnn-actuals", type=Path, default=None,
        help="JSON file with actual synthesis results for GNN comparison",
    )
    args = parser.parse_args()

    predictions: Optional[list[dict]] = None
    actuals: Optional[list[dict]] = None

    if args.gnn_predictions and args.gnn_predictions.exists():
        with args.gnn_predictions.open() as fh:
            predictions = json.load(fh)
    if args.gnn_actuals and args.gnn_actuals.exists():
        with args.gnn_actuals.open() as fh:
            actuals = json.load(fh)

    saved = run_all_plots(args.out_dir, predictions=predictions, actuals=actuals)
    total = sum(len(v) for v in saved.values())
    print(f"Saved {total} plot files to {args.out_dir}")
    for name, paths in saved.items():
        for p in paths:
            print(f"  [{name}] {p}")


if __name__ == "__main__":
    main()
