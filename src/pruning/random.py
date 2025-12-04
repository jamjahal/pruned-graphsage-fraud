"""
Random pruning baseline.

This module assigns random scores to weights and prunes globally to the target sparsity.
It serves as a lower-bound baseline to check if sophisticated pruning methods (SynFlow, Magnitude)
are actually learning useful connectivity.
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
    Compute the global threshold for a desired sparsity from random scores.
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)."

    all_scores = torch.cat([s.flatten() for s in scores.values()])
    num_weights = all_scores.numel()
    num_prune = int(num_weights * sparsity)
    num_keep = max(num_weights - num_prune, 1)

    sorted_scores, _ = torch.sort(all_scores)
    threshold = sorted_scores[-num_keep]
    return threshold


def build_random_masks(model: nn.Module, sparsity: float) -> MaskDict:
    """
    Build binary masks for random pruning at a target sparsity level.
    """
    prunable = _iter_prunable_params(model)
    scores: Dict[PrunableParam, torch.Tensor] = {}

    with torch.no_grad():
        for triplet in prunable:
            _, _, p = triplet
            # Assign random scores in [0, 1)
            scores[triplet] = torch.rand_like(p.data)

    threshold = _global_threshold_from_scores(scores, sparsity)

    masks: MaskDict = {}
    for key, score in scores.items():
        masks[key] = (score >= threshold).to(dtype=torch.float32, device=score.device)
    return masks


def prune_model_random(model: nn.Module, sparsity: float) -> nn.Module:
    """
    Apply random pruning to the model in-place and return it.
    """
    masks = build_random_masks(model, sparsity)
    apply_masks_inplace(model, masks)
    return model

