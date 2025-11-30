"""
Simple magnitude-based pruning baseline.

This module uses the same masking utilities as SynFlow but scores weights
using their absolute values instead of SynFlow's data-agnostic gradient
product. It serves as a control to compare against SynFlow pruning.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.pruning.synflow import (
    MaskDict,
    PrunableParam,
    _iter_prunable_params,
    apply_masks_inplace,
)


def _global_threshold_from_scores(
    scores: Dict[PrunableParam, torch.Tensor], sparsity: float
) -> torch.Tensor:
    """
    Compute the global threshold for a desired sparsity from magnitude scores.
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)."

    all_scores = torch.cat([s.flatten() for s in scores.values()])
    num_weights = all_scores.numel()
    num_prune = int(num_weights * sparsity)
    num_keep = max(num_weights - num_prune, 1)

    sorted_scores, _ = torch.sort(all_scores)
    threshold = sorted_scores[-num_keep]
    return threshold


def build_magnitude_masks(model: nn.Module, sparsity: float) -> MaskDict:
    """
    Build binary masks for magnitude pruning at a target sparsity level.
    """
    prunable = _iter_prunable_params(model)
    scores: Dict[PrunableParam, torch.Tensor] = {}

    with torch.no_grad():
        for triplet in prunable:
            _, _, p = triplet
            scores[triplet] = p.data.abs().clone()

    threshold = _global_threshold_from_scores(scores, sparsity)

    masks: MaskDict = {}
    for key, score in scores.items():
        masks[key] = (score >= threshold).to(dtype=torch.float32, device=score.device)
    return masks


def prune_model_magnitude(model: nn.Module, sparsity: float) -> nn.Module:
    """
    Apply magnitude-based pruning to the model in-place and return it.
    """
    masks = build_magnitude_masks(model, sparsity)
    apply_masks_inplace(model, masks)
    return model


