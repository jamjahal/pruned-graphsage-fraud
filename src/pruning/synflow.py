"""
SynFlow pruning at initialization for GraphSAGE and similar PyTorch models.

This module implements:
- SynFlow score computation (data-agnostic, using all-ones input).
- Global sparsity-based thresholding to create binary masks.
- In-place application of masks to model parameters.
- Utilities to re-enforce masks during training so pruned weights stay zero.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


PrunableParam = Tuple[nn.Module, str, torch.nn.Parameter]
MaskDict = Dict[PrunableParam, torch.Tensor]


def _iter_prunable_params(model: nn.Module) -> Tuple[PrunableParam, ...]:
    """
    Yield (module, name, param) triplets for prunable parameters.

    For now we prune:
    - weights of nn.Linear
    - weights of torch_geometric.nn.SAGEConv
    """
    from torch_geometric.nn import SAGEConv  # lazy import

    prunable: list[PrunableParam] = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, SAGEConv)):
            for name, param in module.named_parameters(recurse=False):
                if name == "weight" and param.requires_grad:
                    prunable.append((module, name, param))
    return tuple(prunable)


def compute_synflow_scores(model: nn.Module, data) -> Dict[PrunableParam, torch.Tensor]:
    """
    Compute SynFlow scores for all prunable parameters.

    Steps:
    - Replace weights with their absolute values.
    - Forward an all-ones input through the network.
    - Backpropagate the sum of outputs.
    - Score for each weight w is |w * grad_w|.
    - Restore original weights afterwards.
    """
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)

    # Save original weights and replace with absolute values.
    prunable = _iter_prunable_params(model)
    original_weights: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for _, _, p in prunable:
            original_weights[id(p)] = p.data.clone()
            p.data = p.data.abs()

    # All-ones input (data-agnostic).
    x_ones = torch.ones_like(data.x, device=device)
    out = model(x_ones, data.edge_index)

    # Sum over all outputs and backpropagate.
    torch.sum(out).backward()

    scores: Dict[PrunableParam, torch.Tensor] = {}
    with torch.no_grad():
        for module, name, p in prunable:
            if p.grad is None:
                continue
            score = (p.data * p.grad).abs().clone()
            scores[(module, name, p)] = score

        # Restore original weights.
        for _, _, p in prunable:
            if id(p) in original_weights:
                p.data = original_weights[id(p)]

    model.zero_grad(set_to_none=True)
    return scores


def _global_threshold_from_scores(
    scores: Dict[PrunableParam, torch.Tensor], sparsity: float
) -> torch.Tensor:
    """
    Compute the global threshold for a desired sparsity.

    `sparsity` is the fraction of weights to set to zero (e.g., 0.9 for 90%).
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)."

    all_scores = torch.cat([s.flatten() for s in scores.values()])
    num_weights = all_scores.numel()
    num_prune = int(num_weights * sparsity)
    num_keep = max(num_weights - num_prune, 1)

    # Keep the largest `num_keep` scores.
    sorted_scores, _ = torch.sort(all_scores)
    threshold = sorted_scores[-num_keep]
    return threshold


def build_synflow_masks(
    model: nn.Module,
    data,
    sparsity: float,
) -> MaskDict:
    """
    Compute SynFlow scores and build binary masks for a target sparsity.
    """
    scores = compute_synflow_scores(model, data)
    threshold = _global_threshold_from_scores(scores, sparsity)

    masks: MaskDict = {}
    for key, score in scores.items():
        masks[key] = (score >= threshold).to(dtype=torch.float32, device=score.device)
    return masks


def apply_masks_inplace(model: nn.Module, masks: MaskDict) -> None:
    """
    Apply masks to the model in-place and register them as buffers so that
    they can be re-applied after optimizer steps.
    """
    with torch.no_grad():
        for (module, name, param), mask in masks.items():
            # Multiply weights by mask.
            param.data.mul_(mask)
            # Register mask as a non-trainable buffer for future enforcement.
            buffer_name = f"{name}_mask"
            # Avoid re-registering if already present.
            if not hasattr(module, buffer_name):
                module.register_buffer(buffer_name, mask)
            else:
                getattr(module, buffer_name).copy_(mask)


def enforce_masks(model: nn.Module) -> None:
    """
    Re-apply all registered masks to ensure pruned weights remain zero.

    This should be called after each optimizer step during training of
    pruned models.
    """
    with torch.no_grad():
        for module in model.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                mask_name = f"{name}_mask"
                if hasattr(module, mask_name):
                    mask = getattr(module, mask_name)
                    param.data.mul_(mask)


def prune_model_synflow(
    model: nn.Module,
    data,
    sparsity: float,
) -> nn.Module:
    """
    Convenience function to:
    - compute SynFlow-based masks
    - apply them to the model in-place

    Returns the same model instance for chaining.
    """
    masks = build_synflow_masks(model, data, sparsity=sparsity)
    apply_masks_inplace(model, masks)
    return model


