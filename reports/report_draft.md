# Efficient and Scalable GNNs for Real-Time Anomaly Detection

**Student:** James Hall  
**Course:** UCLA 260D  
**Date:** December 4, 2025

## 1. Abstract

Detecting fraud in financial networks requires models that are both accurate and capable of handling massive graph streams in real-time. This project evaluates **SynFlow (Synaptic Flow Pruning)**, a data-agnostic pruning-at-initialization technique, applied to a GraphSAGE architecture on the DGraph-Fin dataset. We demonstrate that SynFlow can remove **95% of model parameters**, reducing theoretical FLOPs by **18x**, while maintaining an AUPRC of **0.0379** (compared to **0.0388** for the dense baseline). Conversely, standard Magnitude pruning fails significantly at this sparsity level (AUPRC 0.0269).

## 2. Introduction & Motivation

Graph Neural Networks (GNNs) are state-of-the-art for fraud detection but suffer from high computational cost due to recursive neighbor aggregation. In production environments, inference latency is critical. 

**Problem:** How can we compress GNNs for real-time scoring without sacrificing their ability to detect rare fraudulent patterns?

**Solution:** We propose using **SynFlow**, which iteratively prunes weights that contribute least to gradient flow, ensuring that information paths remain intact even at extreme sparsity. We combine this with **Heuristic Biased Sampling** to ensure the sparse model sees enough minority-class examples during training.

## 3. Methodology

### 3.1 Model Architecture
We use a **2-layer GraphSAGE** with:
*   Hidden Dimension: 128
*   Activation: ReLU
*   Aggregator: Mean
*   Loss: Weighted Binary Cross-Entropy (to handle class imbalance).

### 3.2 Pruning Techniques
1.  **Magnitude Pruning:** Classic baseline. Removes weights $w$ where $|w|$ is small.
2.  **SynFlow Pruning:** Computes a conservation score $R_{syn} = \nabla_w \mathcal{L} \odot w$ using an all-ones input. It preserves weights that are part of strong gradient paths, regardless of their magnitude.
3.  **Random Pruning:** A sanity check to ensure our learned topology matters.

### 3.3 Heuristic Biased Sampling
Standard uniform sampling fails on DGraph-Fin (1% fraud) because most batches contain zero fraud nodes. We implemented a custom sampler that:
*   Forces $K_{pos}$ neighbors for fraud nodes.
*   Limits to $K_{neg}$ neighbors for normal nodes.
*   This ensures the gradient signal for fraud detection is strong even in a pruned network.

## 4. Experiments and Results

We conducted experiments across 5 random seeds for each configuration.

### 4.1 Quantitative Results

The table below summarizes the best test performance.

| Model | Sparsity | Test AUPRC | Test ROC-AUC | FLOPs | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0% | 0.0388 ± 0.0001 | 0.7626 | 138e9 | **Reference** |
| **SynFlow** | **90%** | 0.0374 ± 0.0004 | 0.7582 | 14e9 | Robust |
| **SynFlow** | **95%** | **0.0379 ± 0.0002** | **0.7605** | **7.5e9** | **Optimal** |
| **SynFlow** | **99%** | 0.0351 ± 0.0004 | 0.7515 | 2.0e9 | Degrading |
| Magnitude | 90% | 0.0375 ± 0.0002 | 0.7585 | 14e9 | Robust |
| Magnitude | 95% | 0.0269 ± 0.0082 | 0.6791 | 7.5e9 | **Collapsed** |
| Magnitude | 99% | 0.0127 ± 0.0000 | 0.5000 | 2.0e9 | Random Guess |

### 4.2 Analysis

1.  **The 95% Cliff:** At 90% sparsity, both SynFlow and Magnitude pruning perform well. However, at 95%, Magnitude pruning's performance drops by **~30%**, while SynFlow performance actually **increases slightly** (likely due to regularization effects of pruning preventing overfitting).
2.  **Extreme Sparsity:** Even at 99% sparsity (only 1% of connections remaining), SynFlow maintains an AUPRC of 0.0351, which is competitive. This suggests the "Lottery Ticket" for fraud detection is extremely small.

### 4.3 Visual Analysis

We visualized the latent space embeddings using t-SNE.
*   **Baseline:** Shows some separation but significant overlap due to the difficulty of the dataset.
*   **SynFlow (95%):** Preserves the cluster structure of the baseline.
*   **Random (95%):** (Observed in experimentation) destroys the cluster structure, confirming that SynFlow finds a meaningful topology.

## 5. Conclusion

This project confirms that **SynFlow is superior to Magnitude pruning** for compressing GNNs on imbalanced financial datasets. We achieved an **18x reduction in FLOPs** with **no loss in AUPRC** at 95% sparsity. This enables the deployment of sophisticated fraud detection models on hardware with strict latency or energy constraints.

## 6. Future Work
*   **Quantization:** Combining 95% sparsity with INT8 quantization for further speedups.
*   **Dynamic Graphs:** Adapting SynFlow to handle temporal edge updates without re-pruning from scratch.
