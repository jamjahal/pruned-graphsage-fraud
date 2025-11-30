"""
Convenience launcher to run all main experiment configurations.

Usage:
    python experiments/run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    configs = [
        project_root / "src" / "config" / "baseline.yaml",
        project_root / "src" / "config" / "pruned_magnitude.yaml",
        project_root / "src" / "config" / "pruned_synflow.yaml",
    ]

    for cfg in configs:
        print(f"Running experiments for config: {cfg}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.training.run_experiment",
                "--config",
                str(cfg),
            ],
            cwd=str(project_root),
            check=True,
        )


if __name__ == "__main__":
    main()


