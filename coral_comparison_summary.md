# CORAL ROI Ordinal Baseline Comparison

## Scope and Fairness Constraints

- Fixed backbone: ResNet18.
- Fixed ROI preprocessing style: same crop + resize + normalize flow as existing ROI scripts.
- Fixed data split files: `icdas4_train.csv`, `icdas4_val.csv`, `icdas4_test.csv`.
- Fixed ICDAS4 mapping: `0->0, A->1, B->2, C->3`.
- Only compare output head/loss/decoding; no proposal/gate/two-stage framework changes.

## What Was Added

- New training entry:
  - `train_coral_head_icdas4.py`
- New evaluation entry:
  - `eval_coral_on_roi_icdas4.py`

Additional strict-paper CORAL version (shared-weight + threshold-bias):

- New strict training entry:
  - `train_coral_strict_head_icdas4.py`
- New strict evaluation entry:
  - `eval_coral_strict_on_roi_icdas4.py`

No existing training/evaluation scripts were modified.

## Existing Methods Identified in Repo

- Softmax head on ROI (4-class CE): `train_softmax_head_icdas4.py`
- Existing ordinal heads:
  - Masked ordinal BCE style: `train_ordinal_head_min.py` (`head_type=masked`)
  - Ord2Seq ordinal head: `train_ordinal_head_min.py` (`head_type=ord2seq`) and `ord2seq_head.py`

## CORAL Implementation Notes

- Output dimension is `K-1=3` for ICDAS4 classes (`K=4`).
- Threshold probabilities are:
  - `p_ge1 = p(y > 0)`
  - `p_ge3 = p(y > 1)`
  - `p_ge5 = p(y > 2)`
- Label encoding:
  - `0 -> [0,0,0]`
  - `1 -> [1,0,0]`
  - `2 -> [1,1,0]`
  - `3 -> [1,1,1]`
- Decoding:
  - `pred_class = sum(sigmoid(logits) > 0.5)`

Strict CORAL variant in this repo:

- `train_coral_strict_head_icdas4.py` uses a CORALLayer with:
  - shared weight: one linear projection to scalar logit
  - threshold-specific biases: 3 learnable bias terms for ICDAS4
- Formula: `logits_k = w^T x + b_k`, `k in {1,2,3}`
- This is the strict architecture style referred by CORAL paper implementations.

## Training Command (CORAL)

```bash
python train_coral_head_icdas4.py \
  --train_csv icdas4_train.csv \
  --val_csv icdas4_val.csv \
  --img_root_train <train_image_root> \
  --img_root_val <val_image_root> \
  --img_size 256 \
  --expand 1.25 \
  --bs 64 \
  --epochs 60 \
  --lr 3e-4 \
  --out coral_head_icdas4.pt
```

Training log includes:

- `train_loss`
- `val_loss`
- `val_MAE`
- `val_QWK`

## Evaluation Command (CORAL)

```bash
python eval_coral_on_roi_icdas4.py \
  --val_csv icdas4_val.csv \
  --test_csv icdas4_test.csv \
  --img_root <shared_or_split_specific_image_root> \
  --ckpt coral_head_icdas4.pt \
  --img_size 256 \
  --expand 1.25 \
  --bs 128 \
  --out_val_csv roi_val_icdas4_coral.csv \
  --out_test_csv roi_test_icdas4_coral.csv
```

## Training Command (Strict CORAL)

```bash
python train_coral_strict_head_icdas4.py \
  --train_csv icdas4_train.csv \
  --val_csv icdas4_val.csv \
  --img_root_train <train_image_root> \
  --img_root_val <val_image_root> \
  --img_size 256 \
  --expand 1.25 \
  --bs 64 \
  --epochs 60 \
  --lr 3e-4 \
  --out coral_strict_head_icdas4.pt
```

## Evaluation Command (Strict CORAL)

```bash
python eval_coral_strict_on_roi_icdas4.py \
  --val_csv icdas4_val.csv \
  --test_csv icdas4_test.csv \
  --img_root <shared_or_split_specific_image_root> \
  --ckpt coral_strict_head_icdas4.pt \
  --img_size 256 \
  --expand 1.25 \
  --bs 128 \
  --out_val_csv roi_val_icdas4_coral_strict.csv \
  --out_test_csv roi_test_icdas4_coral_strict.csv
```

Evaluation prints:

- `AUC(>=1)`
- `AUC(>=3)`
- `AUC(>=5)`
- `MAE`
- `QWK`
- confusion matrix

## Output Files

- Model:
  - `coral_head_icdas4.pt`
- Prediction CSVs:
  - `roi_val_icdas4_coral.csv`
  - `roi_test_icdas4_coral.csv`
  - `roi_val_icdas4_coral_strict.csv`
  - `roi_test_icdas4_coral_strict.csv`

Each CSV contains at least:

- `image_id`
- `roi_id`
- `gt_class`
- `pred_class`
- `p_ge1`
- `p_ge3`
- `p_ge5`

## Comparison Protocol

Run six heads under matched settings:

1. Softmax head
2. Your own ordinal head (masked)
3. Existing Ord2Seq ordinal head
4. CORAL head
5. Strict CORAL head
6. DCH-Ordinal head

Then compare on the same val/test CSVs with MAE, QWK, AUC(>=1/3/5), and confusion matrix.

## Current Status

- CORAL scripts are integrated and runnable.
- Latest val/test comparison table has been filled from completed runs.

## Latest Run Results (Same ROI Protocol)

Validation split (`icdas4_val.csv`):

| Method | AUC(>=1) | AUC(>=3) | AUC(>=5) | MAE | QWK |
|---|---:|---:|---:|---:|---:|
| Softmax head (`softmax_head_icdas4.pt`) | 0.812 | 0.906 | 0.890 | 0.314 | 0.614 |
| Your own Ordinal (masked, `ordinal_head_icdas4.pt`) | 0.794 | 0.940 | 0.960 | 0.356 | 0.544 |
| Existing Ordinal (Ord2Seq, `ord2seq_head_icdas4.pt`) | 0.801 | 0.731 | 0.939 | 0.326 | 0.584 |
| CORAL (independent 3-logit) | 0.783 | 0.915 | 0.974 | 0.365 | 0.566 |
| CORAL strict (shared-weight + bias) | 0.822 | 0.897 | 0.945 | 0.410 | 0.573 |
| DCH-Ordinal (hybrid) | 0.808 | 0.912 | 0.942 | 0.336 | 0.596 |

Test split (`icdas4_test.csv`):

| Method | AUC(>=1) | AUC(>=3) | AUC(>=5) | MAE | QWK |
|---|---:|---:|---:|---:|---:|
| Softmax head (`softmax_head_icdas4.pt`) | 0.862 | 0.813 | 0.996 | 0.296 | 0.596 |
| Your own Ordinal (masked, `ordinal_head_icdas4.pt`) | 0.837 | 0.849 | 0.994 | 0.349 | 0.496 |
| Existing Ordinal (Ord2Seq, `ord2seq_head_icdas4.pt`) | 0.857 | 0.680 | 0.998 | 0.286 | 0.568 |
| CORAL (independent 3-logit) | 0.855 | 0.828 | 0.998 | 0.298 | 0.567 |
| CORAL strict (shared-weight + bias) | 0.869 | 0.852 | 0.992 | 0.481 | 0.459 |
| DCH-Ordinal (hybrid) | 0.862 | 0.875 | 0.996 | 0.374 | 0.512 |

Generated CSV files from this run:

- `roi_val_icdas4_softmax_head_icdas4.csv`
- `roi_test_icdas4_softmax_head_icdas4.csv`
- `roi_val_icdas4_ordinal_head_icdas4.csv`
- `roi_test_icdas4_ordinal_head_icdas4.csv`
- `roi_val_icdas4_ord2seq_head_icdas4.csv`
- `roi_test_icdas4_ord2seq_head_icdas4.csv`
- `roi_val_icdas4_coral.csv`
- `roi_test_icdas4_coral.csv`
- `roi_val_icdas4_coral_strict.csv`
- `roi_test_icdas4_coral_strict.csv`
- `roi_val_icdas4_dch_ordinal.csv`
- `roi_test_icdas4_dch_ordinal.csv`

## Threshold Optimization (Val-Tuned, Test-Applied)

To address the mismatch where AUC can improve while MAE/QWK degrades, we did post-hoc threshold tuning on val only:

- Keep model weights fixed.
- Grid-search `(t1, t3, t5)` on val with objective `QWK`.
- Apply the selected thresholds to both val and test for reporting.

Command utility:

```bash
python tools/optimize_icdas4_thresholds_from_roi.py
```

### Best thresholds from validation

| Method | Best `(t1,t3,t5)` |
|---|---|
| Softmax | (0.34, 0.30, 0.18) |
| Your own Ordinal (masked) | (0.78, 0.12, 0.10) |
| Existing Ordinal (Ord2Seq) | (0.62, 0.10, 0.10) |
| CORAL | (0.32, 0.26, 0.10) |
| CORAL strict | (0.38, 0.46, 0.34) |
| DCH-Ordinal | (0.38, 0.34, 0.26) |

### QWK/MAE before vs after tuning

Validation split (`icdas4_val.csv`):

| Method | Base QWK | Tuned QWK | Base MAE | Tuned MAE |
|---|---:|---:|---:|---:|
| Softmax | 0.614 | 0.649 | 0.314 | 0.296 |
| Your own Ordinal (masked) | 0.544 | 0.603 | 0.356 | 0.378 |
| Existing Ordinal (Ord2Seq) | 0.584 | 0.592 | 0.326 | 0.321 |
| CORAL | 0.566 | 0.611 | 0.365 | 0.351 |
| CORAL strict | 0.573 | 0.660 | 0.410 | 0.328 |
| DCH-Ordinal | 0.596 | 0.647 | 0.336 | 0.331 |

Test split (`icdas4_test.csv`):

| Method | Base QWK | Tuned QWK | Base MAE | Tuned MAE |
|---|---:|---:|---:|---:|
| Softmax | 0.596 | 0.591 | 0.296 | 0.303 |
| Your own Ordinal (masked) | 0.496 | 0.528 | 0.349 | 0.408 |
| Existing Ordinal (Ord2Seq) | 0.568 | 0.562 | 0.286 | 0.288 |
| CORAL | 0.567 | 0.552 | 0.298 | 0.324 |
| CORAL strict | 0.459 | 0.552 | 0.481 | 0.401 |
| DCH-Ordinal | 0.512 | 0.613 | 0.374 | 0.349 |

JSON summaries:

- `out_csv/threshold_tuning/softmax_tuning_qwk.json`
- `out_csv/threshold_tuning/masked_ordinal_tuning_qwk.json`
- `out_csv/threshold_tuning/ord2seq_tuning_qwk.json`
- `out_csv/threshold_tuning/coral_tuning_qwk.json`
- `out_csv/threshold_tuning/coral_strict_tuning_qwk.json`
- `out_csv/threshold_tuning/dch_ordinal_tuning_qwk.json`
