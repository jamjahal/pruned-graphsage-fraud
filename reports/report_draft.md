## Efficient and Scalable GNNs for Real-Time Anomaly Detection on DGraph-Fin

### 1. Introduction

- **Motivation**: Real-time detection of rare fraudulent behavior in large financial graphs requires models that are both **expressive** and **computationally efficient**.
- **Challenge**: Standard GNNs suffer from exploding neighborhood sizes and large parameter counts, which hinder low-latency deployment on highly imbalanced datasets such as DGraph-Fin.
- **Goal**: Evaluate whether **data-agnostic pruning (SynFlow)** combined with **heuristic biased sampling** can preserve high anomaly-detection performance while significantly reducing model complexity.

### 2. Related Work

- Briefly survey:
  - GNNs for fraud / anomaly detection on financial and transactional graphs.
  - Network pruning methods, focusing on pruning-at-initialization and SynFlow.
  - Sampling strategies for scalable GNN training (e.g., GraphSAGE neighbor sampling, importance sampling).

### 3. Methodology

#### 3.1 GraphSAGE Baseline

- Node classification on DGraph-Fin using a **GraphSAGE** encoder with:
  - 2–3 layers, hidden dimension 128, ReLU, dropout.
  - Binary fraud / non-fraud labels for nodes.
- **Loss**: Weighted Binary Cross-Entropy with `pos_weight` derived from label frequencies.
- **Metrics**: AUPRC (primary), ROC-AUC, F1, accuracy.

#### 3.2 Pruning Methods

- **Magnitude Pruning**:
  - Score weights by absolute value and globally prune the smallest-magnitude weights to a target sparsity (e.g., 90%).
- **SynFlow Pruning**:
  - Data-agnostic procedure:
    - Replace weights with absolute values.
    - Forward an all-ones input through the network.
    - Backpropagate the sum of outputs and score weights by \\(|w \\cdot \\nabla_w L|\\).
  - Apply a global threshold to achieve the same sparsity level as magnitude pruning.
- Masks are enforced after each optimizer step to maintain sparsity.

#### 3.3 Heuristic Biased Sampling

- **Positive (anomalous) seeds**:
  - Sample **Kpos** neighbors per hop to capture richer local context.
- **Negative (normal) seeds**:
  - Sample **Kneg** neighbors per hop to limit computation on the majority class.
- Implemented as a custom sampler that builds small subgraphs around seeds and exposes a consistent interface to the training loop.

### 4. Experimental Setup

- **Dataset**: DGraph-Fin (v2), with node features, labels, and timestamps.
- **Splits**: Train / validation / test masks; if official splits are unavailable, use stratified random splits that respect the class imbalance.
- **Baselines and Variants**:
  - Baseline GraphSAGE (dense, uniform sampling).
  - Magnitude-pruned GraphSAGE (90% sparse, heuristic sampling).
  - SynFlow-pruned GraphSAGE (90% sparse, heuristic sampling).
- **Training Protocol**:
  - Fixed hyperparameters across seeds for each variant.
  - 5 random seeds for each configuration.
  - Early stopping or fixed-epoch training with selection based on validation AUPRC.

### 5. Results

- **Main Table**:
  - Report mean ± std of best-test AUPRC, ROC-AUC, and F1 across seeds for each variant.
  - Include parameter counts and approximate FLOPs to highlight efficiency gains.
- **Trade-off Analysis**:
  - Discuss how SynFlow and magnitude pruning compare at 90% sparsity.
  - Evaluate the effect of heuristic sampling vs. uniform sampling on recall / AUPRC.
- **Ablations (optional)**:
  - Vary sparsity (e.g., 50%, 90%, 95%) and summarize trends.
  - Vary (Kpos, Kneg) to show robustness of the heuristic sampler.

### 6. Discussion

- Interpret whether SynFlow achieves a better **AUPRC vs. FLOPs** trade-off than magnitude pruning.
- Analyze which components contribute most to performance retention:
  - SynFlow vs. magnitude at equal sparsity.
  - Heuristic sampling vs. uniform sampling at fixed sparsity.
- Reflect on deployment implications:
  - Memory footprint reduction.
  - Expected impact on inference latency in a real-time system.

### 7. Conclusion and Future Work

- Summarize the key empirical findings:
  - To what extent can SynFlow + biased sampling compress GraphSAGE without losing critical anomaly-detection performance?
- Outline future directions:
  - Extending to dynamic or temporal GNNs.
  - Exploring other pruning-at-initialization methods.
  - Investigating more sophisticated sampling or curriculum strategies for extreme imbalance.


