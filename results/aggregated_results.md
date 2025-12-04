| Variant | Sampling | Sparsity | Best Test AUPRC (mean ± std) | Best Test ROC-AUC (mean ± std) | Best Test F1 (mean ± std) | #Params | FLOPs | #Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | uniform | 0.00 | 0.0388 ± 0.0001 | 0.7626 ± 0.0009 | 0.0356 ± 0.0007 | 37505 | 1.385e+11 | 5 |
| pruned_magnitude | heuristic | 0.90 | 0.0375 ± 0.0002 | 0.7585 ± 0.0012 | 0.0348 ± 0.0006 | 37505 | 1.441e+10 | 5 |
| pruned_magnitude | heuristic | 0.95 | 0.0269 ± 0.0082 | 0.6791 ± 0.0900 | 0.0262 ± 0.0024 | 37505 | 7.518e+09 | 5 |
| pruned_magnitude | heuristic | 0.99 | 0.0127 ± 0.0000 | 0.5000 ± 0.0000 | 0.0250 ± 0.0000 | 37505 | 2.004e+09 | 5 |
| pruned_random | heuristic | 0.90 | 0.0370 ± 0.0008 | 0.7566 ± 0.0033 | 0.0344 ± 0.0002 | 37505 | 1.441e+10 | 5 |
| pruned_random | heuristic | 0.99 | 0.0159 ± 0.0039 | 0.5439 ± 0.0500 | 0.0250 ± 0.0000 | 37505 | 2.004e+09 | 3 |
| pruned_synflow | heuristic | 0.90 | 0.0374 ± 0.0004 | 0.7582 ± 0.0007 | 0.0350 ± 0.0004 | 37505 | 1.441e+10 | 5 |
| pruned_synflow | heuristic | 0.95 | 0.0379 ± 0.0002 | 0.7605 ± 0.0010 | 0.0350 ± 0.0004 | 37505 | 7.518e+09 | 5 |
| pruned_synflow | heuristic | 0.99 | 0.0351 ± 0.0004 | 0.7515 ± 0.0013 | 0.0342 ± 0.0010 | 37505 | 2.004e+09 | 5 |
| pruned_synflow | heuristic | 1.00 | 0.0265 ± 0.0037 | 0.7100 ± 0.0130 | 0.0269 ± 0.0024 | 37505 | 7.643e+08 | 5 |
