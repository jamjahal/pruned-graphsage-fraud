"""
Efficiency metrics: parameter counts and rough FLOPs estimates.

The FLOPs estimates are approximate and tailored for GraphSAGE-style
message passing, but they are sufficient to capture relative changes
across pruned vs. dense variants.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_graphsage_flops(
    model: nn.Module,
    num_nodes: int,
    num_edges: int,
) -> Optional[float]:
    """
    Roughly estimate FLOPs for a forward pass of a GraphSAGE-like model.

    Assumptions:
    - Each SAGEConv layer performs:
      - Aggregation: O(E * d) operations.
      - Transformation: O(N * d^2) operations (linear layer).
    - We sum across all SAGEConv layers with the same hidden dimension.
    - We ignore activation functions and dropout for simplicity.

    If the model does not follow this pattern, returns None.
    """
    from torch_geometric.nn import SAGEConv  # imported lazily

    conv_layers = [m for m in model.modules() if isinstance(m, SAGEConv)]
    if not conv_layers:
        return None

    # Assume constant hidden dimension across layers (typical for GraphSAGE).
    hidden_dim = conv_layers[0].out_channels

    flops = 0.0
    for _ in conv_layers:
        flops += num_edges * hidden_dim  # aggregation
        flops += num_nodes * (hidden_dim * hidden_dim)  # transformation

    # Final linear projection if present.
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += num_nodes * (m.in_features * m.out_features)

    return float(flops)


