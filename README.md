## Scalable Pruned GraphSAGE for Real-Time Fraud Detection (260D Project)

This repository implements the project described in `documentation/260D Project Proposals.md`:

- **Model**: GraphSAGE-based GNN for anomaly / fraud detection
- **Techniques**: SynFlow pruning (at initialization) + magnitude pruning baseline + heuristic biased neighbor sampling
- **Dataset**: DGraph-Fin (large-scale financial fraud graph)

The goal is to build a **deployable, efficient, and high-recall** graph model suitable for real-time applications.

### Repository Layout

- `data/` – DGraph-Fin raw/processed data (not tracked in git; keep large files here).
- `src/`
  - `data/` – dataset loading and preprocessing (e.g., `dgraph_fin.py`).
  - `models/` – GraphSAGE and pruned variants.
  - `sampling/` – heuristic biased neighbor sampler.
  - `pruning/` – SynFlow and magnitude pruning utilities.
  - `training/` – training loops, experiment runners.
  - `metrics/` – AUPRC and efficiency metrics (FLOPs / parameters).
  - `config/` – experiment configuration files (YAML/JSON).
- `experiments/` – scripts to launch and aggregate experiments.
- `reports/` – paper-style report drafts, figures, and tables.
- `docs/` – notes, design documents, and presentation materials.

### Environment Setup

1. **Create / activate virtual environment**

```bash
cd /Users/jameshall/UCLA/260D/project
source venv/bin/activate  # or your preferred venv name
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**

- Place the DGraph-Fin archive under `data/` (already present as `DGraphFin2.zip`).
- Later steps in `src/data/dgraph_fin.py` will handle extracting / loading this data.

### Running the Baseline (once implemented)

After the baseline GraphSAGE model and training script are implemented, you will be able to run:

```bash
python -m src.training.train_baseline --config src/config/baseline.yaml
```

This will:

- Load DGraph-Fin
- Train an unpruned GraphSAGE model with uniform neighbor sampling
- Log AUPRC and other metrics

### Reproducibility and Experiments

The `experiments/` directory will contain:

- Scripts to run baseline, magnitude-pruned, and SynFlow-pruned models
- Utilities to aggregate metrics across seeds and generate tables/plots

Refer to `documentation/260D Project Proposals.md` and `reports/` for detailed methodology and analysis once experiments are complete.


