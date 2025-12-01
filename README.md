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
python3.10 -m venv venv  # if not already created
source venv/bin/activate
```

2. **Install Python dependencies (PyTorch + DGL + utilities)**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

On Google Colab, you can run:

```python
!pip install torch==2.0.0 torchvision==0.15.1 torchaudio==0.15.1
!pip install dgl==2.4.0
!pip install -r requirements.txt
```

3. **Prepare the dataset**

- Place the DGraph-Fin archive under `data/` (already present as `DGraphFin2.zip`).
- The loader in `src/data/dgraph_fin.py` will read from the unpacked npz/npy files.

### Running the Baseline

Launch a baseline run (uniform sampling, default hyperparameters):

```bash
python -m src.training.train_baseline --data-root data/DGraphFin2 --sampling uniform --device cpu
```

Key CLI flags mirror the `TrainConfig` dataclass. For example, to switch to the heuristic sampler:

```bash
python -m src.training.train_baseline --sampling heuristic --k-pos 25 --k-neg 5
```

Both commands will:

- Load DGraph-Fin via the new DGL-based loader
- Train an unpruned GraphSAGE model with the requested mini-batch sampler
- Log AUPRC/ROC-AUC and other metrics each epoch

### Reproducibility and Experiments

The `experiments/` directory contains configs and scripts for:

- Baseline, magnitude-pruned, and SynFlow-pruned models
- Utilities to aggregate metrics across seeds and generate tables/plots

Refer to `documentation/260D Project Proposals.md` and `reports/` for detailed methodology and analysis once experiments are complete.


