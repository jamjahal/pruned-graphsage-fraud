| Variant | Sampling | Sparsity | Best Test AUPRC (mean ± std) | Best Test ROC-AUC (mean ± std) | Best Test F1 (mean ± std) | #Params | FLOPs | #Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | uniform | 0.00 | 0.0388 ± 0.0001 | 0.7626 ± 0.0009 | 0.0356 ± 0.0007 | 37505 | 1.385e+11 | 5 |
| pruned_magnitude | heuristic | 0.90 | 0.0375 ± 0.0002 | 0.7585 ± 0.0012 | 0.0348 ± 0.0006 | 37505 | 1.441e+10 | 5 |
| pruned_synflow | heuristic | 0.90 | 0.0374 ± 0.0004 | 0.7582 ± 0.0007 | 0.0350 ± 0.0004 | 37505 | 1.441e+10 | 5 |
