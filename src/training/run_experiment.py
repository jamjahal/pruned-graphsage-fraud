"""
Unified experiment runner for baseline, magnitude-pruned, and SynFlow-pruned models.

This script:
- Loads an experiment config from YAML.
- Runs training for a specified list of random seeds.
- Logs per-epoch metrics to stdout.
- Saves summary JSON files (best validation metrics and corresponding test metrics).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import torch
import yaml
from torch import nn, optim

from src.data.dgraph_fin import load_dgraphfin_dataset
from src.metrics.efficiency import count_parameters, estimate_graphsage_flops
from src.metrics.metrics import compute_binary_metrics
from src.models.graphsage import build_graphsage_for_data
from src.pruning.magnitude import prune_model_magnitude
from src.pruning.synflow import enforce_masks, prune_model_synflow
from src.training.train_baseline import (
    TrainConfig as BaseTrainConfig,
    build_loaders,
    compute_pos_weight,
    set_seed,
)


@dataclass
class TrainHyperparams:
    epochs: int
    lr: float
    weight_decay: float
    hidden_channels: int
    num_layers: int
    dropout: float
    batch_size: int
    num_neighbors: List[int]
    num_hops: int
    k_pos: int
    k_neg: int
    device: str


@dataclass
class ExperimentConfig:
    data_root: str
    output_dir: str
    model_variant: str  # baseline, pruned_magnitude, pruned_synflow
    sampling: str       # uniform, heuristic
    sparsity: float
    train: TrainHyperparams
    seeds: List[int]


def load_experiment_config(path: str) -> ExperimentConfig:
    """
    Load an experiment description from YAML.

    Parses the config file at `path`, instantiates the nested dataclasses
    (`TrainHyperparams`, `ExperimentConfig`), and returns a fully typed
    object that downstream helpers can consume.
    """
    with open(path, "r") as f:
        cfg_raw = yaml.safe_load(f)

    train_raw = cfg_raw["train"]
    train = TrainHyperparams(
        epochs=train_raw["epochs"],
        lr=train_raw["lr"],
        weight_decay=train_raw["weight_decay"],
        hidden_channels=train_raw["hidden_channels"],
        num_layers=train_raw["num_layers"],
        dropout=train_raw["dropout"],
        batch_size=train_raw["batch_size"],
        num_neighbors=list(train_raw["num_neighbors"]),
        num_hops=train_raw["num_hops"],
        k_pos=train_raw["k_pos"],
        k_neg=train_raw["k_neg"],
        device=train_raw["device"],
    )

    return ExperimentConfig(
        data_root=cfg_raw["data_root"],
        output_dir=cfg_raw["output_dir"],
        model_variant=cfg_raw["model_variant"],
        sampling=cfg_raw["sampling"],
        sparsity=float(cfg_raw["sparsity"]),
        train=train,
        seeds=list(cfg_raw["seeds"]),
    )


def _build_base_train_config(exp_cfg: ExperimentConfig, seed: int) -> BaseTrainConfig:
    """
    Convert the high-level experiment config into the TrainConfig used by training code.

    Copies shared hyperparameters (epochs, hidden size, sampling settings, etc.)
    and injects the specific random seed for the current run.
    """
    t = exp_cfg.train
    return BaseTrainConfig(
        data_root=exp_cfg.data_root,
        epochs=t.epochs,
        lr=t.lr,
        weight_decay=t.weight_decay,
        hidden_channels=t.hidden_channels,
        num_layers=t.num_layers,
        dropout=t.dropout,
        batch_size=t.batch_size,
        num_neighbors=t.num_neighbors,
        num_hops=t.num_hops,
        sampling=exp_cfg.sampling,
        k_pos=t.k_pos,
        k_neg=t.k_neg,
        device=t.device,
        seed=seed,
    )


def run_single_seed(exp_cfg: ExperimentConfig, seed: int) -> dict:
    """
    Run training + evaluation for one random seed and return a summary.

    * Builds the dataset/model,
    * applies pruning if requested,
    * trains for the configured number of epochs,
    * tracks the best validation AUPRC, and
    * returns a dictionary with metrics (val/test), parameter counts, FLOPs, and bookkeeping info.
    """
    base_cfg = _build_base_train_config(exp_cfg, seed)
    set_seed(base_cfg.seed)

    device = torch.device(base_cfg.device)
    dataset = load_dgraphfin_dataset(root=exp_cfg.data_root)
    data = dataset[0].to(device)

    model = build_graphsage_for_data(
        data,
        hidden_channels=base_cfg.hidden_channels,
        num_layers=base_cfg.num_layers,
        dropout=base_cfg.dropout,
    ).to(device)

    # Apply pruning according to the model variant.
    if exp_cfg.model_variant == "pruned_magnitude":
        model = prune_model_magnitude(model, sparsity=exp_cfg.sparsity)
    elif exp_cfg.model_variant == "pruned_synflow":
        model = prune_model_synflow(model, data=data, sparsity=exp_cfg.sparsity)
    elif exp_cfg.model_variant != "baseline":
        raise ValueError(f"Unknown model_variant: {exp_cfg.model_variant}")

    # Efficiency metrics.
    num_params = count_parameters(model, trainable_only=True)
    approx_flops = estimate_graphsage_flops(
        model, num_nodes=data.num_nodes, num_edges=data.num_edges
    )
    print(
        f"[Seed {seed}] Variant: {exp_cfg.model_variant} | "
        f"Params: {num_params} | FLOPs: {approx_flops:.3e}" if approx_flops is not None else ""
    )

    pos_weight = compute_pos_weight(data.y.to(device)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=base_cfg.lr, weight_decay=base_cfg.weight_decay)

    train_loader = build_loaders(data, base_cfg)

    best_val_auprc = float("-inf")
    best_epoch = -1
    best_val_metrics = {}
    best_test_metrics = {}

    for epoch in range(1, base_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits_all = model(batch.x, batch.edge_index).view(-1)
            if hasattr(batch, "seed_idx"):
                seed_idx = batch.seed_idx.to(logits_all.device)
            else:
                seed_idx = torch.arange(batch.batch_size, device=logits_all.device)

            logits = logits_all[seed_idx]
            targets = batch.y[seed_idx].float()

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            # Enforce masks for pruned models so zeros remain zeros.
            if exp_cfg.model_variant in ("pruned_magnitude", "pruned_synflow"):
                enforce_masks(model)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        avg_loss = total_loss / max(total_examples, 1)

        # Validation + test metrics.
        model.eval()
        with torch.no_grad():
            logits_full = model(data.x, data.edge_index).view(-1)

            def _eval_mask(mask: torch.Tensor) -> dict:
                mask_logits = logits_full[mask]
                mask_targets = data.y[mask].float()
                loss_val = criterion(mask_logits, mask_targets.to(device)).item()
                probs_val = torch.sigmoid(mask_logits)
                m = compute_binary_metrics(mask_targets.cpu(), probs_val.cpu())
                m["loss"] = loss_val
                return m

            val_metrics = _eval_mask(data.val_mask)
            test_metrics = _eval_mask(data.test_mask)

        print(
            f"[Seed {seed}] Epoch {epoch:03d} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUPRC: {val_metrics['auprc']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics

    summary = {
        "seed": seed,
        "model_variant": exp_cfg.model_variant,
        "sampling": exp_cfg.sampling,
        "sparsity": exp_cfg.sparsity,
        "num_params": num_params,
        "approx_flops": approx_flops,
        "best_epoch": best_epoch,
        "best_val": best_val_metrics,
        "best_test": best_test_metrics,
        "final_val_auprc": float(val_metrics["auprc"]),
        "final_test_auprc": float(test_metrics["auprc"]),
    }
    return summary


def save_summary(output_dir: str, seed: int, summary: dict) -> None:
    """
    Persist a single-seed summary to disk as JSON.

    Ensures `output_dir` exists, writes the summary to `seed_<n>.json`,
    and logs where it was saved so downstream aggregation scripts can read it.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary for seed {seed} to {path}")


def main() -> None:
    """
    CLI entry point for the experiment runner.

    Parses the `--config` argument, loads the experiment description,
    iterates over all requested seeds, and saves their summaries.
    """
    parser = argparse.ArgumentParser(description="Run GNN pruning experiments from YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    args = parser.parse_args()

    exp_cfg = load_experiment_config(args.config)

    for seed in exp_cfg.seeds:
        summary = run_single_seed(exp_cfg, seed)
        save_summary(exp_cfg.output_dir, seed, summary)


if __name__ == "__main__":
    main()


