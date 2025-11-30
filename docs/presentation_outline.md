## 260D Project Presentation Outline

### Slide 1–2: Title and Motivation

- Project title, team members, course, and term.
- Real-world motivation: large-scale financial fraud detection and the need for efficient GNNs.

### Slide 3–4: Problem and Dataset

- Problem statement: real-time anomaly detection on massive, imbalanced graphs.
- Overview of DGraph-Fin (nodes, edges, feature types, anomaly ratio).

### Slide 5–7: Methodology

- GraphSAGE baseline architecture and training setup.
- Pruning methods:
  - Magnitude pruning at initialization.
  - SynFlow pruning at initialization (data-agnostic).
- Heuristic biased neighbor sampling (Kpos / Kneg, intuition, and diagram).

### Slide 8–10: Experiments and Results

- Experimental setup: baselines, variants, seeds, metrics.
- Key tables/plots:
  - Best Test AUPRC / ROC-AUC / F1 for each variant.
  - AUPRC vs. parameter count / FLOPs.
- Highlight whether SynFlow + heuristic sampling meets the deployment-style efficiency goals.

### Slide 11–12: Discussion and Takeaways

- Interpretation of main trends:
  - SynFlow vs. magnitude pruning.
  - Effectiveness of heuristic sampling on recall / AUPRC.
- Limitations and assumptions (e.g., static graph, single dataset).
- Key takeaways for deploying scalable GNNs in real-world systems.

### Slide 13: Future Work and Q&A

- Extensions to temporal/dynamic GNNs, alternative pruning schemes, and richer samplers.
- Invite questions and discussion.


