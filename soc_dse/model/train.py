"""
GNN training script.

Splits the NetlistDataset 70/15/15 (train/val/test), trains with MSE loss,
logs per-target MAE and R², and saves the best validation checkpoint to
``model/checkpoints/best.pt``.

Usage
-----
    python -m soc_dse.model.train [options]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from soc_dse.model.dataset import NetlistDataset, PPA_KEYS, GRAPHS_DIR
from soc_dse.model.gnn import PPAGNN

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
CHECKPOINT_DIR: Path = _ROOT / "checkpoints"
BEST_CHECKPOINT: Path = CHECKPOINT_DIR / "best.pt"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-target mean absolute error, shape [num_outputs]."""
    return (pred - target).abs().mean(dim=0)


def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-target R² (coefficient of determination), shape [num_outputs].
    R² = 1 − SS_res / SS_tot
    """
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
    return 1.0 - ss_res / ss_tot.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: PPAGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean MSE loss."""
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: PPAGNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluate on a DataLoader.

    Returns
    -------
    (mse_loss, mae_per_target, r2_per_target)
    """
    model.eval()
    criterion = nn.MSELoss()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        total_loss += criterion(pred, batch.y).item() * batch.num_graphs
        all_preds.append(pred.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mse = total_loss / len(loader.dataset)
    return mse, compute_mae(preds, targets), compute_r2(preds, targets)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    graphs_dir: Path = GRAPHS_DIR,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    dropout_p: float = 0.3,
    seed: int = 42,
) -> dict[str, float]:
    """
    Train the GNN on the netlist dataset.

    Returns a dict with final test metrics:
    ``test_mse``, ``test_mae_<key>``, ``test_r2_<key>`` for each PPA target.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    # ---- Dataset -----------------------------------------------------------
    dataset = NetlistDataset(graphs_dir=graphs_dir)
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    if n_test < 1:
        raise ValueError(
            f"Dataset too small ({n} samples) for 70/15/15 split. "
            "Need at least 7 graphs."
        )

    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train_set = dataset[indices[:n_train].tolist()]
    val_set   = dataset[indices[n_train:n_train + n_val].tolist()]
    test_set  = dataset[indices[n_train + n_val:].tolist()]

    log.info("Split: train=%d  val=%d  test=%d", n_train, n_val, n_test)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ---- Model + optimiser --------------------------------------------------
    num_node_features = dataset[0].x.shape[1]
    model = PPAGNN(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
    ).to(device)
    log.info("Model parameters: %d", model.num_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ---- Training loop ------------------------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "num_node_features": num_node_features,
                    "hidden_dim": hidden_dim,
                    "dropout_p": dropout_p,
                },
                BEST_CHECKPOINT,
            )

        if epoch % 10 == 0 or epoch == 1:
            r2_str = "  ".join(
                f"{k}={val_r2[i]:.3f}" for i, k in enumerate(PPA_KEYS)
            )
            log.info(
                "Epoch %3d | train_loss=%.4f  val_loss=%.4f | R² %s",
                epoch, train_loss, val_loss, r2_str,
            )

    # ---- Final test evaluation -----------------------------------------------
    checkpoint = torch.load(BEST_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_mae, test_r2 = evaluate(model, test_loader, device)

    log.info("=== Test results ===")
    metrics: dict[str, float] = {"test_mse": round(test_loss, 6)}
    for i, key in enumerate(PPA_KEYS):
        metrics[f"test_mae_{key}"] = round(test_mae[i].item(), 6)
        metrics[f"test_r2_{key}"] = round(test_r2[i].item(), 6)
        log.info("  %s  MAE=%.4f  R²=%.4f", key, test_mae[i], test_r2[i])

    mean_r2 = test_r2.mean().item()
    log.info("  Mean R²=%.4f  (target ≥0.75)", mean_r2)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train GNN for PPA prediction")
    parser.add_argument("--graphs-dir", type=Path, default=GRAPHS_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train(
        graphs_dir=args.graphs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout,
        seed=args.seed,
    )
    print("\nFinal test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
