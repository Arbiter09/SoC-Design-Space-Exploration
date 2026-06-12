"""
GraphSAGE-based GNN for PPA prediction.

Architecture
------------
- 3-layer GraphSAGE encoder (node_features → hidden_dim → hidden_dim → hidden_dim)
- Global mean pooling  →  graph-level embedding  [batch, hidden_dim]
- 2-layer MLP regression head  →  [batch, 3]  (area, delay, power)
- Dropout applied after each layer (enabled at train time; kept ON for MC Dropout
  at inference time to produce confidence intervals)

MC Dropout inference
--------------------
Call ``model.enable_mc_dropout()`` before running forward passes to keep
dropout stochastic even when the model is in eval mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class PPAGNN(nn.Module):
    """
    GraphSAGE network predicting [area, delay, power_estimate] from a netlist graph.

    Parameters
    ----------
    num_node_features:
        Dimension of input node feature vectors (default: 19).
    hidden_dim:
        Width of all hidden layers (default: 64).
    num_outputs:
        Number of regression targets (default: 3).
    dropout_p:
        Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        num_node_features: int = 19,
        hidden_dim: int = 64,
        num_outputs: int = 3,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout_p = dropout_p
        self._mc_dropout = False  # set True to force stochastic dropout in eval mode

        # GraphSAGE encoder
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        # Batch normalisation after each conv
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # MLP regression head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

    # ------------------------------------------------------------------
    # MC Dropout control
    # ------------------------------------------------------------------

    def enable_mc_dropout(self) -> None:
        """Force dropout to remain stochastic during eval (for MC Dropout CI)."""
        self._mc_dropout = True

    def disable_mc_dropout(self) -> None:
        """Restore standard eval-time behaviour (dropout disabled)."""
        self._mc_dropout = False

    def _dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout, honouring the MC Dropout flag."""
        if self.training or self._mc_dropout:
            return F.dropout(x, p=self.dropout_p, training=True)
        return x

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:          Node features  [N, num_node_features]
        edge_index: COO edges      [2, E]
        batch:      Batch vector   [N]  (assigned by PyG DataLoader)

        Returns
        -------
        torch.Tensor  shape [B, num_outputs]
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self._dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self._dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self._dropout(x)

        # Graph-level pooling
        x = global_mean_pool(x, batch)  # [B, hidden_dim]

        # Regression head
        return self.mlp(x)  # [B, num_outputs]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
