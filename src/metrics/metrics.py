"""
Metric utilities for binary node classification on graphs.

Primary metric: AUPRC (Area Under the Precision-Recall Curve),
with additional ROC-AUC, accuracy, F1, precision, and recall.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_numpy_binary(
    y_true: torch.Tensor | np.ndarray, y_score: torch.Tensor | np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    y_true = y_true.reshape(-1)
    y_score = y_score.reshape(-1)
    return y_true, y_score


def compute_binary_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_score: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a suite of binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels (0/1).
    y_score : array-like
        Predicted probabilities or logits (will be converted to probabilities
        via sigmoid if they appear to be logits).
    threshold : float
        Decision threshold for metrics that require hard predictions.
    """
    y_true_np, y_score_np = _to_numpy_binary(y_true, y_score)

    # If scores are outside [0, 1], treat them as logits and apply sigmoid.
    if y_score_np.min() < 0.0 or y_score_np.max() > 1.0:
        y_score_np = 1.0 / (1.0 + np.exp(-y_score_np))

    y_pred_np = (y_score_np >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    try:
        metrics["auprc"] = float(average_precision_score(y_true_np, y_score_np))
    except ValueError:
        metrics["auprc"] = float("nan")

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_score_np))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    metrics["accuracy"] = float(accuracy_score(y_true_np, y_pred_np))

    # Some metrics may fail when only one class is present.
    for name, fn in [
        ("f1", f1_score),
        ("precision", precision_score),
        ("recall", recall_score),
    ]:
        try:
            metrics[name] = float(fn(y_true_np, y_pred_np, zero_division=0))
        except ValueError:
            metrics[name] = float("nan")

    return metrics


