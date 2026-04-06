# ICDAS4 ROI Stability Summary (3 Seeds)

## Protocol
- Same split/preprocess/backbone/training-budget across methods.
- Fixed threshold decoding at default settings (no per-method post-hoc tuning).
- No proposal-stage changes.
- ResNet18 backbone and existing method-specific scripts are reused.
- Only random seed is varied.
- Seeds: 2026, 2027, 2028

## Table 1: Test stability leaderboard
| Rank | Method | MAE (mean±std) | QWK (mean±std) | AUC>=1 | AUC>=3 | AUC>=5 |
|---|---|---|---|---|---|---|
| 1 | Ord2Seq-guided Softmax OrdPlus | 0.308±0.023 | 0.589±0.031 | 0.869±0.012 | 0.863±0.003 | 0.966±0.040 |
| 2 | Softmax baseline | 0.335±0.033 | 0.577±0.027 | 0.877±0.009 | 0.866±0.004 | 0.965±0.050 |
| 3 | Existing Ordinal (Ord2Seq) | 0.328±0.022 | 0.571±0.027 | 0.870±0.004 | 0.738±0.070 | 0.968±0.016 |
| 4 | CORN | 0.335±0.032 | 0.545±0.039 | 0.853±0.010 | 0.850±0.026 | 0.996±0.002 |

## Table 2: Validation stability leaderboard
| Rank | Method | MAE (mean±std) | QWK (mean±std) | AUC>=1 | AUC>=3 | AUC>=5 |
|---|---|---|---|---|---|---|
| 1 | Ord2Seq-guided Softmax OrdPlus | 0.327±0.026 | 0.651±0.022 | 0.808±0.005 | 0.918±0.010 | 0.962±0.028 |
| 2 | Existing Ordinal (Ord2Seq) | 0.347±0.006 | 0.598±0.039 | 0.809±0.007 | 0.795±0.040 | 0.934±0.003 |
| 3 | Softmax baseline | 0.344±0.020 | 0.584±0.029 | 0.793±0.003 | 0.914±0.005 | 0.952±0.040 |
| 4 | CORN | 0.343±0.017 | 0.582±0.037 | 0.807±0.005 | 0.916±0.012 | 0.974±0.023 |

## Table 3: Main method vs Softmax baseline
| Method | Test MAE mean±std | Test QWK mean±std | ΔQWK vs Softmax | ΔMAE vs Softmax |
|---|---|---|---|---|
| Ord2Seq-guided Softmax OrdPlus | 0.308±0.023 | 0.589±0.031 | +0.012 | -0.027 |
| Softmax baseline | 0.335±0.033 | 0.577±0.027 | +0.000 | +0.000 |

## Brief Stability Notes
- Largest test QWK fluctuation is CORN with std=0.039.
- Largest test MAE fluctuation is Softmax baseline with std=0.033.
