# Proposal 1: 

# Pruning Scalable GNNs for Real-Time Fraud Detection: The Efficacy of Data-Agnostic Compression and Heuristic Biased Sampling

---

This project will demonstrate a practical and resource-efficient path for deploying high-quality GNNs on large, imbalanced graph data. The main contribution is proving that the combination of **data-agnostic model compression (SynFlow)** with **data-aware heuristic sampling** yields an efficient and robust model that is suitable for real-time applications, thereby addressing core challenges in large-scale machine learning systems.

## **Project Proposal: Efficient and Scalable GNNs for Real-Time Anomaly Detection**

### 

## **Evaluating the Efficacy of Data-Agnostic Pruning (SynFlow) and Heuristic Biased Sampling for Deployable, High-Recall Graph Neural Networks in Large-Scale Imbalanced Networks**

## ---

## Problem Statement and Motivation

The detection of rare, complex anomalies (e.g., collusion rings, sophisticated fraud) requires the use of **Graph Neural Networks (GNNs)** to model relationships across massive, high-dimensional datasets. However, deploying these models is constrained by two major computational challenges:

1. **Scaling Data Volume (Complexity):** Training GNNs on graphs with billions of nodes and edges is computationally intractable because the standard aggregation process leads to an exponential explosion of neighbors, requiring vast memory and processing time.  
2. **Scaling Model Size (Efficiency):** The resulting large GNN models are too slow for **real-time inference** and cannot be easily deployed in low-latency production environments.  
3. **Data Imbalance (Robustness):** The target anomalies are extremely rare ($\\ll 1\\%$ of observations). Uniform sampling methods fail to adequately capture the minority signal, leading to models with poor **recall** on the critical class.

## ---

## Proposed Solution and Methodology

We propose an innovative, two-part methodology that addresses complexity, efficiency, and robustness simultaneously, resulting in a highly compact and effective GNN.

### A. Model Compression for Efficiency (SynFlow)

The goal is to dramatically reduce the model footprint while preserving its capacity for anomaly detection.

* **Technique:** **SynFlow Pruning**. This is a **pruning-at-initialization** method that identifies and prunes the network's least important weights **before training begins**. It is **data-agnostic**, relying only on the network structure to calculate weight criticality.  
* **Method:** We will target a high sparsity level (e.g., **90% parameter reduction**) on a standard GNN architecture (e.g., GraphSAGE).  
* **Hypothesis:** SynFlow will identify a highly sparse subnetwork that retains most of the full model's performance while achieving critical gains in efficiency (reduced latency).

### B. Data Scaling and Imbalance Mitigation (Biased Sampling)

The goal is to enable efficient training of the GraphSAGE architecture while ensuring the model learns from the rare anomaly signals.

* **Technique:** **Heuristic Biased Sampling (Modified GraphSAGE)**. This technique integrates an intelligent sampling strategy into the GNN's data loading phase to correct for label imbalance.  
* **Method:** When constructing the training mini-batches:  
  1. **For Anomaly Nodes (Positive Label):** **Oversample** the neighborhood (sample Kpos neighbors, where Kpos is large). This ensures the GNN aggregates the full context of the rare anomalies.  
  2. **For Normal Nodes (Negative Label):** **Undersample** the neighborhood (sample  Knegneighbors, where Kneg  is small). This reduces computational overhead on the massive majority class.  
* **Complementary Method:** A **Weighted Binary Cross-Entropy Loss** could be applied during training to further prioritize correct classification of the minority class.

# 

# 

# 

# Experimental Design and Evaluation

## Datasets

We will use large, publicly available graph anomaly datasets 

* DGraph-Fin, or similar structured datasets) to ensure rigorous and reproducible benchmarking.

Below are a couple of candidate datasets that are publicly available.

| Dataset Name | Domain / Structure | Scale (Approx.) | Imbalance Ratio | Relevance to Project |
| :---- | :---- | :---- | :---- | :---- |
| [**DGraph-Fin**](https://dgraph.xinye.com/dataset) | **Financial/Loan Guarantor Network** | \~3.7M Nodes, 4.3M Edges | \~1.3% Anomalous | **Highest Relevance.** Designed specifically for Graph Anomaly Detection (GAD) in a financial context. Highly cited in recent GNN research. Original paper [link](https://arxiv.org/abs/2207.03579) |
| **Elliptic** | **Cryptocurrency Payment Flow** | \~200k Nodes, 234k Edges | \~9.8% Illicit | Excellent representation of a **transaction graph** with clear labels (illicit vs. legal). Highly heterogeneous. |
| [**IEEE-CIS Fraud Detection**](https://www.kaggle.com/competitions/ieee-fraud-detection) **(Kaggle)** | **E-commerce/Transaction Features** | \~400k Transactions | \~0.17% Fraudulent | **Excellent Imbalance.** Requires preprocessing to convert transactions/identities into a **bipartite graph** (customer-merchant) to use GNNs. |
| **Credit Card Fraud Detection (Kaggle)** | **Credit Card Transactions** | \~285k Transactions | \~0.172% Fraudulent | **Extreme Imbalance.** Requires feature engineering (data is PCA-transformed) and graph construction, but provides a classic, highly imbalanced fraud benchmark. |
| **YelpChi / Amazon** | **Reviewer Interactions** | \~46k Nodes / 12k Nodes | \~10-15% Anomalous | Good for testing GNN methods on **reviewer/entity fraud**, but smaller scale than the financial datasets. |

#### **B. Baselines and Comparisons**

| Model Name | Pruning Method | Sampling Method | Purpose |
| :---- | :---- | :---- | :---- |
| **Baseline** | None (Unpruned) | Uniform Sampling | Establishes the performance ceiling and provides initial model size/latency metrics. |
| **SynFlow (Test)** | **SynFlow (90% Sparse)** | **Heuristic Biased Sampling** | The primary experiment—testing the combined efficiency and robustness. |
| **Magnitude (Control)** | Simple Magnitude Pruning | Heuristic Biased Sampling | Controls for the superior performance of the SynFlow technique over simpler pruning. |

## 

## Metrics: 

**Robustness Metric (Primary):** **AUPRC** (Area Under the Precision-Recall Curve). This is the standard metric for highly imbalanced classification problems, focusing specifically on the performance on the minority class.  
**Efficiency Metrics (Secondary):**

* **Parameter Reduction:** The percentage decrease in trainable weights (e.g., 90%).  
* **FLOPs Reduction:** Projected reduction in floating-point operations, indicative of decreased **inference latency**.

## ---

# Proposed breakdown of project components:

Split up between team members

| Phase | Duration (Suggested) | Work Package (Task) | Owner Role | Parallel Opportunities | Deliverable |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **I: Setup & Baseline** | 1 Week | **1\. Data Setup** | Allan | Runs parallel to 2 & 3\. | Clean, loadable torch\_geometric or DGL Graph object with train/val/test masks. |
|  |  | **2\. Baseline Architecture** |  | Runs parallel to 1 & 3\. | Unpruned **GraphSAGE** model code initialized and ready to train. |
|  |  | **3\. Metrics & Loss** |  | Runs parallel to 1 & 2\. | **Weighted Loss** function implementation; **AUPRC** and **FLOPs** calculation utilities. |
| **II: Core Implementation** | 2 Weeks | **4\. SynFlow Pruning Logic** | Allan | **CRITICAL PATH.** Requires output from 2\. Runs parallel to 5\. | Function that applies **SynFlow** score, creates a 90% sparsity mask, and generates the sparse model. |
|  |  | **5\. Pruned Model Training (Test)** |  | Runs parallel to 4\. Uses output from 3\. | Training loop adapted to use the **sparse model mask** and **Weighted Loss**. |
| **III: Experiment & Analysis** | 2 Weeks | **6\. Hyperparameter Search & Tuning** |  | Runs parallel to 7\. | Optimized hyperparameters for the **Final Pruned Model** and **Baseline**. |
|  |  | **7\. Final Experiment Runs** |  | Runs parallel to 6\. | Automated scripts to run all 3 scenarios (Baseline, Pruned-Mag, Pruned-SynFlow) 5 times each. |
|  |  | **8\. Results Analysis & Write-up** |  | Starts once 7 is complete. | Final Report/Paper: Tables comparing **AUPRC Retention** vs. **Parameter/FLOPs Reduction**. |

# Questions / Concerns / Risks:

