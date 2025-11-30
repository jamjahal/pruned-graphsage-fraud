"""
Baseline GraphSAGE model for node-level fraud / anomaly detection.

This implementation uses PyTorch Geometric's `SAGEConv` layers and produces a
single logit per node (for binary classification with BCEWithLogitsLoss).
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    Simple GraphSAGE encoder followed by a linear projection to a single logit.

    Parameters
    ----------
    in_channels: int
        Dimensionality of node features.
    hidden_channels: int
        Hidden dimensionality of intermediate GraphSAGE layers.
    num_layers: int
        Total number of GraphSAGE layers (>= 2 recommended).
    dropout: float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.out_proj = nn.Linear(hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute node logits.

        Parameters
        ----------
        x: Tensor, shape [num_nodes, in_channels]
        edge_index: LongTensor, shape [2, num_edges]

        Returns
        -------
        logits: Tensor, shape [num_nodes, 1]
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        logits = self.out_proj(x)
        return logits


def build_graphsage_for_data(
    data, hidden_channels: int = 128, num_layers: int = 2, dropout: float = 0.2
) -> GraphSAGE:
    """
    Convenience constructor that infers `in_channels` from `data.x`.
    """
    in_channels = data.num_node_features
    return GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )


