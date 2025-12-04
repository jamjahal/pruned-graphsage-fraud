"""
Plot helper for aggregated experiment results.

After running experiments and `aggregate_results.py`, call:
    python experiments/plot_results.py

This will read `aggregated_results.json` and emit:
1. Bar chart: Best Test AUPRC per variant.
2. Scatter plot: AUPRC vs FLOPs trade-off.
3. Line plot: Sparsity vs AUPRC (comparing pruning methods).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "results"
    agg_path = results_root / "aggregated_results.json"

    if not agg_path.exists():
        print(f"Aggregated results not found at {agg_path}. Run aggregate_results.py first.")
        return

    with open(agg_path, "r") as f:
        rows = json.load(f)

    variants = [r["model_variant"] for r in rows]
    sparsities = [r["sparsity"] for r in rows]
    
    # Generate readable labels including sparsity
    labels = []
    for v, s in zip(variants, sparsities):
        if v == "baseline":
            labels.append("Baseline")
        else:
            # e.g. "pruned_synflow" -> "SynFlow"
            clean_v = v.replace("pruned_", "").capitalize()
            # e.g. 0.95 -> 95%
            labels.append(f"{clean_v} ({s*100:.0f}%)")

    auprc_means = [r["best_test_auprc_mean"] for r in rows]
    auprc_stds = [r["best_test_auprc_std"] for r in rows]
    flops = [r["approx_flops"] for r in rows]

    # -------------------------------------------------------------------------
    # 1. Bar Chart: AUPRC per Variant
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, auprc_means, yerr=auprc_stds, capsize=4, alpha=0.7)
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel("Best Test AUPRC")
    plt.title("Best Test AUPRC by Variant")
    plt.tight_layout()
    plt.savefig(results_root / "auprc_by_variant.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Scatter: AUPRC vs FLOPs
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for l, a, f in zip(labels, auprc_means, flops):
        if f is None:
            continue
        plt.scatter(f, a, label=l, s=100, alpha=0.8)
        # Offset text slightly to avoid overlap
        plt.annotate(l, (f, a), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xscale("log")
    plt.xlabel("Approximate FLOPs (log scale)")
    plt.ylabel("Best Test AUPRC")
    plt.title("Efficiency Frontier: AUPRC vs FLOPs")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(results_root / "auprc_vs_flops.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Line Plot: Sparsity vs AUPRC
    # -------------------------------------------------------------------------
    # Organize data by method (SynFlow, Magnitude, Random)
    # Key: method_name -> [(sparsity, auprc, std)]
    methods: Dict[str, List] = {}
    baseline_auprc = 0.0
    baseline_std = 0.0

    for r in rows:
        v = r["model_variant"]
        s = r["sparsity"]
        a = r["best_test_auprc_mean"]
        std = r["best_test_auprc_std"]

        if v == "baseline":
            baseline_auprc = a
            baseline_std = std
            continue

        # Parse method name (e.g., "pruned_synflow" -> "SynFlow")
        if "synflow" in v:
            label = "SynFlow"
        elif "magnitude" in v:
            label = "Magnitude"
        elif "random" in v:
            label = "Random"
        else:
            label = v

        if label not in methods:
            methods[label] = []
        methods[label].append((s, a, std))

    plt.figure(figsize=(8, 6))
    
    # Plot methods lines
    colors = {"SynFlow": "#C44E52", "Magnitude": "#4C72B0", "Random": "#55A868"}
    
    for label, points in methods.items():
        # Sort by sparsity
        points.sort(key=lambda p: p[0])
        sparsities = [p[0] for p in points]
        scores = [p[1] for p in points]
        stds = [p[2] for p in points]
        
        color = colors.get(label, "gray")
        plt.errorbar(sparsities, scores, yerr=stds, label=label, marker='o', capsize=4, color=color, linewidth=2)

    # Plot Baseline Horizontal Line
    if baseline_auprc > 0:
        plt.axhline(y=baseline_auprc, color='black', linestyle='--', label=f'Baseline (Dense) {baseline_auprc:.4f}', alpha=0.6)
        # Optional: Add shaded region for baseline std
        plt.axhspan(baseline_auprc - baseline_std, baseline_auprc + baseline_std, color='black', alpha=0.1)

    plt.xlabel("Sparsity (Fraction of Weights Pruned)")
    plt.ylabel("Best Test AUPRC")
    plt.title("Robustness to Pruning: Sparsity vs AUPRC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0.0) # Ensure 0 is visible if scores are low, or adjust to min
    
    # Invert x-axis? No, standard is 0 -> 1 (High sparsity on right)
    # But focus on the high end (0.5 - 1.0)?
    # Let's keep it auto, but ensure limits make sense
    
    plt.tight_layout()
    plt.savefig(results_root / "sparsity_vs_auprc.png", dpi=200)
    plt.close()

    print(f"Saved plots under {results_root}")


if __name__ == "__main__":
    main()
