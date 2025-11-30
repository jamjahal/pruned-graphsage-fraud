"""
Plot helper for aggregated experiment results.

After running experiments and `aggregate_results.py`, call:
    python experiments/plot_results.py

This will read `aggregated_results.json` and emit simple plots comparing
Best Test AUPRC across variants and the trade-off between AUPRC and FLOPs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    """
    Plot helper for aggregated experiment results.

    After running the training suite and `aggregate_results.py`, execute:
        python experiments/plot_results.py

    This script reads `experiments/results/aggregated_results.json` and produces
    two publication-ready figures:
    - A bar chart (with error bars) comparing best test AUPRC across variants.
    - A scatter plot showing the AUPRC vs FLOPs trade-off on a log scale.

    Both figures are saved back into `experiments/results/` so they can be
    dropped directly into the report or presentation deck.
    """
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "experiments" / "results"
    agg_path = results_root / "aggregated_results.json"

    if not agg_path.exists():
        print(f"Aggregated results not found at {agg_path}. Run aggregate_results.py first.")
        return

    with open(agg_path, "r") as f:
        rows = json.load(f)

    variants = [r["model_variant"] for r in rows]
    auprc_means = [r["best_test_auprc_mean"] for r in rows]
    auprc_stds = [r["best_test_auprc_std"] for r in rows]
    flops = [r["approx_flops"] for r in rows]

    # Bar chart: AUPRC per variant.
    plt.figure(figsize=(6, 4))
    x = range(len(variants))
    plt.bar(x, auprc_means, yerr=auprc_stds, capsize=4)
    plt.xticks(x, variants, rotation=30)
    plt.ylabel("Best Test AUPRC")
    plt.title("Best Test AUPRC by Variant")
    plt.tight_layout()
    plt.savefig(results_root / "auprc_by_variant.png", dpi=200)

    # Scatter: AUPRC vs FLOPs.
    plt.figure(figsize=(6, 4))
    for v, a, f in zip(variants, auprc_means, flops):
        if f is None:
            continue
        plt.scatter(f, a, label=v)
        plt.text(f, a, v)
    plt.xscale("log")
    plt.xlabel("Approximate FLOPs (log scale)")
    plt.ylabel("Best Test AUPRC")
    plt.title("AUPRC vs FLOPs Trade-off")
    plt.tight_layout()
    plt.savefig(results_root / "auprc_vs_flops.png", dpi=200)

    print(f"Saved plots under {results_root}")


if __name__ == "__main__":
    main()


