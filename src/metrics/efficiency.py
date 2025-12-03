"""
Efficiency metrics: parameter counts and rough FLOPs estimates.

The FLOPs estimates are approximate and tailored for GraphSAGE-style
message passing, but they are sufficient to capture relative changes
across pruned vs. dense variants.
"""

from __future__ import annotations

from typing import Optional

from dgl.nn import SAGEConv
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
    - Aggregation: O(E * d_in) operations (dense, unpruned).
    - Transformation: O(N * NNZ(W)) operations where NNZ is non-zero weights.
    """
    flops = 0.0
    
    # Track if we found any recognized layers
    found_layers = False

    for m in model.modules():
        if isinstance(m, SAGEConv):
            found_layers = True
            # Aggregation cost (Edge message passing)
            # Usually assumes source features are dense: E * in_feats
            # This part is generally NOT pruned in weight pruning
            in_feats = m._in_src_feats
            flops += num_edges * in_feats

            # Transformation cost (Linear projection of neighbors/self)
            # SAGEConv usually has a linear layer 'fc_neigh' and possibly 'fc_self'
            # We inspect its sub-modules or parameters to find the Linear layers
            # DGL SAGEConv uses `fc_neigh` and `fc_self` which are nn.Linear
            for name, child in m.named_children():
                if isinstance(child, nn.Linear):
                    # Count non-zero weights for transformation
                    w = child.weight
                    nnz = torch.count_nonzero(w).item()
                    # Linear layer is x @ W.T + b
                    # Ops: N * NNZ (if we treat it as sparse-dense mul)
                    flops += num_nodes * nnz

        elif isinstance(m, nn.Linear):
            # Standalone linear layers (e.g. final classifier not inside SAGEConv)
            # Check if this linear layer is NOT part of a SAGEConv we already counted
            is_submodule = False
            for parent in model.modules():
                if isinstance(parent, SAGEConv) and m in parent.modules():
                    is_submodule = True
                    break
            
            if not is_submodule:
                found_layers = True
                nnz = torch.count_nonzero(m.weight).item()
                flops += num_nodes * nnz

    if not found_layers:
        return None

    return float(flops)
