"""
Aggregate per-seed JSON summaries into variant-level tables.

Run after experiments have completed:
    python experiments/aggregate_results.py

This will scan `experiments/results/*/seed_*.json` and produce:
- A Markdown table summarizing mean / std metrics per variant.
- A JSON file with the aggregated statistics for further analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_seed_summaries(results_dir: Path) -> List[dict]:
    """
    Read all `seed_*.json` summaries inside `results_dir`.

    Returns a list of per-seed dictionaries (one per training run) so they
    can be aggregated into variant-level statistics.
    """
    summaries: List[dict] = []
    for path in sorted(results_dir.glob("seed_*.json")):
        with open(path, "r") as f:
            summaries.append(json.load(f))
    return summaries


def aggregate_variant(summaries: List[dict]) -> Dict[str, object]:
    """
    Collapse multiple seed summaries for the same variant into summary stats.

    Computes mean/std for best-test metrics (AUPRC, ROC-AUC, F1) and copies
    shared metadata such as sparsity, sampling mode, parameter count, etc.
    """
    if not summaries:
        return {}

    variant = summaries[0]["model_variant"]
    sampling = summaries[0]["sampling"]
    sparsity = summaries[0]["sparsity"]
    num_params = summaries[0]["num_params"]
    approx_flops = summaries[0]["approx_flops"]

    def collect(path: List[str]) -> np.ndarray:
        vals = []
        for s in summaries:
            d = s
            for key in path:
                d = d[key]
            vals.append(d)
        return np.asarray(vals, dtype=float)

    best_test_auprc = collect(["best_test", "auprc"])
    best_test_roc = collect(["best_test", "roc_auc"])
    best_test_f1 = collect(["best_test", "f1"])

    return {
        "model_variant": variant,
        "sampling": sampling,
        "sparsity": sparsity,
        "num_params": num_params,
        "approx_flops": approx_flops,
        "best_test_auprc_mean": float(best_test_auprc.mean()),
        "best_test_auprc_std": float(best_test_auprc.std(ddof=0)),
        "best_test_roc_auc_mean": float(best_test_roc.mean()),
        "best_test_roc_auc_std": float(best_test_roc.std(ddof=0)),
        "best_test_f1_mean": float(best_test_f1.mean()),
        "best_test_f1_std": float(best_test_f1.std(ddof=0)),
        "num_seeds": len(summaries),
    }


def main() -> None:
    """
    Entry point for results aggregation.

    Iterates over every variant directory in `experiments/results`, loads seed
    summaries, writes an aggregated JSON file plus a Markdown table for easy
    inclusion in reports, and logs where the outputs were saved.
    """
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "experiments" / "results"

    aggregated: List[Dict[str, object]] = []
    for variant_dir in sorted(results_root.glob("*")):
        if not variant_dir.is_dir():
            continue
        summaries = load_seed_summaries(variant_dir)
        if not summaries:
            continue
        agg = aggregate_variant(summaries)
        agg["results_dir"] = str(variant_dir)
        aggregated.append(agg)

    if not aggregated:
        print("No result summaries found under experiments/results.")
        return

    # Save JSON.
    out_json = results_root / "aggregated_results.json"
    with open(out_json, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Wrote aggregated JSON to {out_json}")

    # Save Markdown table.
    out_md = results_root / "aggregated_results.md"
    headers = [
        "Variant",
        "Sampling",
        "Sparsity",
        "Best Test AUPRC (mean ± std)",
        "Best Test ROC-AUC (mean ± std)",
        "Best Test F1 (mean ± std)",
        "#Params",
        "FLOPs",
        "#Seeds",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in aggregated:
        line = "| {variant} | {sampling} | {sparsity:.2f} | {auprc_mean:.4f} ± {auprc_std:.4f} | {roc_mean:.4f} ± {roc_std:.4f} | {f1_mean:.4f} ± {f1_std:.4f} | {params} | {flops} | {seeds} |".format(
            variant=row["model_variant"],
            sampling=row["sampling"],
            sparsity=row["sparsity"],
            auprc_mean=row["best_test_auprc_mean"],
            auprc_std=row["best_test_auprc_std"],
            roc_mean=row["best_test_roc_auc_mean"],
            roc_std=row["best_test_roc_auc_std"],
            f1_mean=row["best_test_f1_mean"],
            f1_std=row["best_test_f1_std"],
            params=row["num_params"],
            flops=f"{row['approx_flops']:.3e}" if row["approx_flops"] is not None else "N/A",
            seeds=row["num_seeds"],
        )
        lines.append(line)

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote aggregated Markdown table to {out_md}")


if __name__ == "__main__":
    main()


