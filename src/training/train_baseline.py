"""
Baseline training script for an unpruned GraphSAGE model on DGraph-Fin.

This script:
- Loads the DGraph-Fin dataset.
- Builds a GraphSAGE model.
- Trains with weighted BCEWithLogitsLoss to handle class imbalance.
- Uses uniform neighbor sampling for mini-batch training.

Later experiment runners can import and reuse `train_baseline` programmatically.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import nn, optim
from torch_geometric.loader import NeighborLoader

from src.data.dgraph_fin import load_dgraphfin_dataset
from src.models.graphsage import build_graphsage_for_data
from src.metrics.efficiency import count_parameters, estimate_graphsage_flops
from src.metrics.metrics import compute_binary_metrics
from src.sampling.heuristic_sampler import HeuristicNeighborLoader


def _detect_default_device() -> str:
    """
    Choose a sensible default device:
    - Prefer Apple Metal (MPS) on Apple Silicon when available.
    - Otherwise prefer CUDA if available.
    - Fall back to CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainConfig:
    data_root: str = "data/DGraphFin2"  # directory containing processed npy files
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-5
    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 2048
    num_neighbors: Sequence[int] = (25, 25)
    num_hops: int = 2
    sampling: str = "uniform"  # or "heuristic"
    k_pos: int = 25
    k_neg: int = 5
    device: str = field(default_factory=_detect_default_device)
    seed: int = 42


def set_seed(seed: int) -> None:
    """
    Seed Python/PyTorch RNGs (CPU + CUDA) for reproducible training runs.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """
    Derive the positive-class weight for BCEWithLogitsLoss from label counts.

    Returns neg/pos so the minority class gets upweighted; defaults to 1.0 if
    no positives are present to avoid division by zero.
    """
    y = y.view(-1)
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
    if pos == 0:
        # Avoid division by zero; fall back to no weighting.
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / max(pos, 1), dtype=torch.float32)


def build_loaders(data, cfg: TrainConfig):
    """
    Construct the appropriate NeighborLoader/Heuristic loader for training.

    Uses the train mask to pick seed nodes, then instantiates either the built-in
    uniform sampler or the custom heuristic sampler based on `cfg.sampling`.
    """
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    if cfg.sampling == "uniform":
        train_loader = NeighborLoader(
            data,
            num_neighbors=list(cfg.num_neighbors),
            batch_size=cfg.batch_size,
            input_nodes=train_idx,
            shuffle=True,
        )
    elif cfg.sampling == "heuristic":
        train_loader = HeuristicNeighborLoader(
            data=data,
            input_nodes=train_idx,
            batch_size=cfg.batch_size,
            num_hops=cfg.num_hops,
            k_pos=cfg.k_pos,
            k_neg=cfg.k_neg,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {cfg.sampling}")

    return train_loader


def train_baseline(cfg: Optional[TrainConfig] = None) -> None:
    """
    Train the baseline GraphSAGE model end-to-end using the provided config.

    Handles:
    * dataset loading
    * model creation
    * weighted loss/optimizer setup,
    * mini-batch training (uniform or heuristic sampling),
    * validation/test metrics computed on the full graph.
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    dataset = load_dgraphfin_dataset(root=cfg.data_root)
    data = dataset[0].to(device)

    model = build_graphsage_for_data(
        data,
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    # Efficiency metrics (parameters, FLOPs) for logging.
    num_params = count_parameters(model, trainable_only=True)
    approx_flops = estimate_graphsage_flops(
        model, num_nodes=data.num_nodes, num_edges=data.num_edges
    )
    print(f"Model parameters (trainable): {num_params}")
    if approx_flops is not None:
        print(f"Approximate FLOPs per forward pass: {approx_flops:.3e}")

    pos_weight = compute_pos_weight(data.y.to(device)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = build_loaders(data, cfg)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits_all = model(batch.x, batch.edge_index).view(-1)

            # Determine which nodes are seeds for this batch.
            if hasattr(batch, "seed_idx"):
                seed_idx = batch.seed_idx.to(logits_all.device)
            else:
                # For NeighborLoader, the first `batch.batch_size` nodes are seeds.
                seed_idx = torch.arange(batch.batch_size, device=logits_all.device)

            logits = logits_all[seed_idx]
            targets = batch.y[seed_idx].float()

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        avg_loss = total_loss / max(total_examples, 1)
        # Validation metrics using full-graph forward on the validation mask.
        model.eval()
        with torch.no_grad():
            logits_full = model(data.x, data.edge_index).view(-1)
            val_mask = data.val_mask
            val_logits = logits_full[val_mask]
            val_targets = data.y[val_mask].float()
            val_loss = criterion(val_logits, val_targets.to(device)).item()

            # Convert logits to probabilities and compute metrics.
            probs = torch.sigmoid(val_logits)
            metrics = compute_binary_metrics(val_targets.cpu(), probs.cpu())

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUPRC: {metrics['auprc']:.4f} | "
            f"Val ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    # Final evaluation on validation and test sets.
    model.eval()
    with torch.no_grad():
        logits_full = model(data.x, data.edge_index).view(-1)

        def _eval_mask(mask: torch.Tensor, name: str) -> None:
            mask_logits = logits_full[mask]
            mask_targets = data.y[mask].float()
            loss_val = criterion(mask_logits, mask_targets.to(device)).item()
            probs_val = torch.sigmoid(mask_logits)
            m = compute_binary_metrics(mask_targets.cpu(), probs_val.cpu())
            print(
                f"{name} | Loss: {loss_val:.4f} | "
                f"AUPRC: {m['auprc']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | "
                f"F1: {m['f1']:.4f}"
            )

        _eval_mask(data.val_mask, "Validation final")
        _eval_mask(data.test_mask, "Test final")


def parse_args() -> argparse.Namespace:
    """
    CLI argument parser: exposes TrainConfig knobs for ad-hoc runs.

    Populates defaults from TrainConfig so `python -m src.training.train_baseline`
    can be tuned without editing the source.
    """
    parser = argparse.ArgumentParser(description="Train baseline GraphSAGE on DGraph-Fin.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/DGraphFin2",
        help="Root directory for DGraph-Fin data (processed npy files).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        default=[25, 25],
        help="Number of neighbors to sample at each layer.",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=2,
        help="Number of sampling hops for heuristic sampler.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="uniform",
        choices=["uniform", "heuristic"],
        help="Neighbor sampling strategy.",
    )
    parser.add_argument(
        "--k-pos",
        type=int,
        default=25,
        help="Number of neighbors for positive-labeled seeds (heuristic sampler).",
    )
    parser.add_argument(
        "--k-neg",
        type=int,
        default=5,
        help="Number of neighbors for negative-labeled seeds (heuristic sampler).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_detect_default_device(),
        help="Torch device string (cpu, cuda, or mps). Default auto-detects.",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """
    Entry point for the standalone baseline training script.

    Parses CLI args, builds a TrainConfig, and calls `train_baseline`.
    """
    args = parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbors,
        num_hops=args.num_hops,
        sampling=args.sampling,
        k_pos=args.k_pos,
        k_neg=args.k_neg,
        device=args.device,
        seed=args.seed,
    )
    train_baseline(cfg)


if __name__ == "__main__":
    main()


