"""
Baseline GraphSAGE model for node-level fraud / anomaly detection.

This implementation uses DGL's `SAGEConv` layers and produces a
single logit per node (for binary classification with BCEWithLogitsLoss).
"""

from __future__ import annotations

from typing import Sequence, Union

import dgl
import torch
from torch import nn
from dgl.nn import SAGEConv


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
        aggregator_type: str = "mean",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggregator_type))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggregator_type))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.out_proj = nn.Linear(hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reinitialize all learnable weights/biases in the module.

        Calls `reset_parameters()` on each SAGEConv layer and applies
        Xavier/zero init to the output projection so the model starts
        from a clean, reproducible state before training or evaluation.
        """
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, g: Union[dgl.DGLGraph, Sequence[dgl.DGLGraph]], x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute node logits.

        Parameters
        ----------
        g: Either a full DGLGraph or a sequence of DGL blocks (one per layer).
        x: Node features corresponding to the graph input. For block-based
           mini-batch training, this should be the feature tensor of
           `blocks[0].srcdata["feat"]`.

        Returns
        -------
        logits: Tensor, shape [num_nodes, 1]
        """
        if isinstance(g, (list, tuple)):
            h = x
            for conv, block in zip(self.convs, g):
                h = conv(block, h)
                h = self.activation(h)
                h = self.dropout(h)
        else:
            h = x
            for conv in self.convs:
                h = conv(g, h)
                h = self.activation(h)
                h = self.dropout(h)

        logits = self.out_proj(h)
        return logits


def build_graphsage_for_data(
    data,
    hidden_channels: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    aggregator_type: str = "mean",
) -> GraphSAGE:
    """
    Convenience constructor that infers `in_channels` from graph node features.
    """
    # `data` is expected to be a DGLGraph with node features stored under 'feat'.
    x = data.ndata["feat"]
    in_channels = x.size(-1)
    return GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        aggregator_type=aggregator_type,
    )


