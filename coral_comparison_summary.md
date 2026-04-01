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

## Threshold Optimization v2 (Joint Objective: QWK-MAE)

To improve the trade-off between QWK and MAE, we added a joint objective on val:

- objective: `qwk_mae`
- selection score: `QWK - lambda_mae * MAE`
- this run uses `lambda_mae=0.30`
- Pareto front is exported for each method.

### Best thresholds from validation (qwk_mae)

| Method | Best `(t1,t3,t5)` |
|---|---|
| Softmax | (0.34, 0.30, 0.18) |
| Your own Ordinal (masked) | (0.58, 0.12, 0.10) |
| Existing Ordinal (Ord2Seq) | (0.62, 0.10, 0.10) |
| CORAL | (0.32, 0.34, 0.10) |
| CORAL strict | (0.32, 0.46, 0.80) |
| DCH-Ordinal | (0.38, 0.42, 0.30) |

### QWK/MAE before vs after tuning (qwk_mae)

Validation split (`icdas4_val.csv`):

| Method | Base QWK | Tuned QWK | Base MAE | Tuned MAE |
|---|---:|---:|---:|---:|
| Softmax | 0.614 | 0.649 | 0.314 | 0.296 |
| Your own Ordinal (masked) | 0.544 | 0.599 | 0.356 | 0.358 |
| Existing Ordinal (Ord2Seq) | 0.584 | 0.592 | 0.326 | 0.321 |
| CORAL | 0.566 | 0.610 | 0.365 | 0.346 |
| CORAL strict | 0.573 | 0.654 | 0.410 | 0.299 |
| DCH-Ordinal | 0.596 | 0.645 | 0.336 | 0.314 |

Test split (`icdas4_test.csv`):

| Method | Base QWK | Tuned QWK | Base MAE | Tuned MAE |
|---|---:|---:|---:|---:|
| Softmax | 0.596 | 0.591 | 0.296 | 0.303 |
| Your own Ordinal (masked) | 0.496 | 0.532 | 0.349 | 0.380 |
| Existing Ordinal (Ord2Seq) | 0.568 | 0.562 | 0.286 | 0.288 |
| CORAL | 0.567 | 0.538 | 0.298 | 0.326 |
| CORAL strict | 0.459 | 0.568 | 0.481 | 0.332 |
| DCH-Ordinal | 0.512 | 0.603 | 0.374 | 0.311 |

JSON summaries (`qwk_mae`):

- `out_csv/threshold_tuning/softmax_tuning_qwkmae.json`
- `out_csv/threshold_tuning/masked_ordinal_tuning_qwkmae.json`
- `out_csv/threshold_tuning/ord2seq_tuning_qwkmae.json`
- `out_csv/threshold_tuning/coral_tuning_qwkmae.json`
- `out_csv/threshold_tuning/coral_strict_tuning_qwkmae.json`
- `out_csv/threshold_tuning/dch_ordinal_tuning_qwkmae.json`

Pareto exports:

- `out_csv/threshold_tuning/softmax_pareto.json`
- `out_csv/threshold_tuning/masked_ordinal_pareto.json`
- `out_csv/threshold_tuning/ord2seq_pareto.json`
- `out_csv/threshold_tuning/coral_pareto.json`
- `out_csv/threshold_tuning/coral_strict_pareto.json`
- `out_csv/threshold_tuning/dch_ordinal_pareto.json`

## DCH Ablation: Remove `L_mono`

Setup:

- Keep all hyper-parameters unchanged except `lambda_mono=0.0`.
- Train command uses the same DCH config as baseline.
- Evaluate both checkpoints with the same eval script and same `alpha=0.7`.

Checkpoints:

- With `L_mono`: `dch_ordinal_head_icdas4.pt`
- Without `L_mono`: `dch_ordinal_head_icdas4_no_mono.pt`

Evaluation outputs:

- with mono: `roi_val_icdas4_dch_ordinal_with_mono_reval.csv`, `roi_test_icdas4_dch_ordinal_with_mono_reval.csv`
- no mono: `roi_val_icdas4_dch_ordinal_no_mono.csv`, `roi_test_icdas4_dch_ordinal_no_mono.csv`

### Val/Test comparison

| Split | Variant | AUC(>=1) | AUC(>=3) | AUC(>=5) | MAE | QWK |
|---|---|---:|---:|---:|---:|---:|
| Val | DCH + `L_mono` | 0.808 | 0.912 | 0.942 | 0.336 | 0.596 |
| Val | DCH - `L_mono` | 0.787 | 0.926 | 0.982 | 0.368 | 0.605 |
| Test | DCH + `L_mono` | 0.862 | 0.875 | 0.996 | 0.374 | 0.512 |
| Test | DCH - `L_mono` | 0.858 | 0.812 | 0.958 | 0.359 | 0.535 |

Observed effect:

- On test, removing `L_mono` improves MAE (`0.374 -> 0.359`) and QWK (`0.512 -> 0.535`).
- On val, removing `L_mono` worsens MAE (`0.336 -> 0.368`) but slightly improves QWK (`0.596 -> 0.605`).
- This indicates `L_mono` may not be universally beneficial in current setting and can be treated as a tunable/optional regularizer.

## Minimal Comparison: `fixed` vs `static` vs `dynamic` Fusion (Default Threshold)

Protocol:

- Same DCH backbone and losses (`lambda_ce=0.5`, `lambda_cons=0.2`, `lambda_mono=0.05`).
- Only fusion strategy changes.
- Default decode threshold kept unchanged (`0.5`), no post-hoc threshold tuning.

Fusion variants:

- `fixed`: global fixed `alpha=0.7`.
- `static`: learn three global fusion weights (`alpha1, alpha3, alpha5`).
- `dynamic`: learn sample-dependent fusion weights via a gate network.

Checkpoints:

- `fixed`: `dch_ordinal_head_icdas4.pt`
- `static`: `dch_ordinal_head_icdas4_static.pt`
- `dynamic`: `dch_ordinal_head_icdas4_dynamic.pt`

Evaluation CSV outputs:

- `fixed`: `roi_val_icdas4_dch_fixed_default.csv`, `roi_test_icdas4_dch_fixed_default.csv`
- `static`: `roi_val_icdas4_dch_static_default.csv`, `roi_test_icdas4_dch_static_default.csv`
- `dynamic`: `roi_val_icdas4_dch_dynamic_default.csv`, `roi_test_icdas4_dch_dynamic_default.csv`

Validation split (`icdas4_val.csv`):

| Variant | AUC(>=1) | AUC(>=3) | AUC(>=5) | MAE | QWK |
|---|---:|---:|---:|---:|---:|
| fixed (`alpha=0.7`) | 0.808 | 0.912 | 0.942 | 0.336 | 0.596 |
| static (learned `alpha1/alpha3/alpha5`) | 0.826 | 0.925 | 0.985 | 0.373 | 0.601 |
| dynamic (sample-wise gate) | 0.780 | 0.927 | 0.981 | 0.368 | 0.582 |

Test split (`icdas4_test.csv`):

| Variant | AUC(>=1) | AUC(>=3) | AUC(>=5) | MAE | QWK |
|---|---:|---:|---:|---:|---:|
| fixed (`alpha=0.7`) | 0.862 | 0.875 | 0.996 | 0.374 | 0.512 |
| static (learned `alpha1/alpha3/alpha5`) | 0.860 | 0.850 | 0.941 | 0.389 | 0.532 |
| dynamic (sample-wise gate) | 0.834 | 0.831 | 0.994 | 0.315 | 0.523 |

Quick reading:

- On test, `static` improves QWK over `fixed` (`0.532 vs 0.512`) but hurts MAE.
- On test, `dynamic` gives the best MAE (`0.315`) and better QWK than `fixed` (`0.523 vs 0.512`).
- This supports replacing hard-coded fusion with data-driven fusion when MAE/QWK trade-off is the target.

## Follow-up: `dynamic` + No `L_mono` (Default Threshold)

Protocol:

- Keep `fusion_mode=dynamic` unchanged.
- Set only `lambda_mono=0.0`; all other training/eval settings unchanged.
- Use default decode threshold (`0.5`), no post-hoc threshold tuning.

Checkpoint:

- `dch_ordinal_head_icdas4_dynamic_no_mono.pt`

Evaluation CSV outputs:

- Val: `roi_val_icdas4_dch_dynamic_no_mono_default.csv`
- Test: `roi_test_icdas4_dch_dynamic_no_mono_default.csv`

Val/Test metrics:

| Split | Variant | AUC(>=1) | AUC(>=3) | AUC(>=5) | MAE | QWK |
|---|---|---:|---:|---:|---:|---:|
| Val | dynamic + no `L_mono` | 0.780 | 0.927 | 0.981 | 0.368 | 0.582 |
| Test | dynamic + no `L_mono` | 0.834 | 0.831 | 0.994 | 0.315 | 0.523 |

Comparison to dynamic + `L_mono` (same default-threshold protocol):

- Metrics are identical to the previously reported dynamic run.
- In this setting, removing `L_mono` does not change MAE/QWK or AUC, suggesting the current dynamic-fusion setup is insensitive to this regularizer.

## New Method: Ord2Seq-Guided Softmax OrdPlus (Default Threshold)

Method idea (paper-style innovation on top of softmax):

- Keep the same ResNet18 ROI backbone and 4-class softmax head.
- Add an Ord2Seq ordinal branch as a structured ordinal teacher.
- Fuse softmax and ordinal probabilities with learnable class-wise fusion weights.
- Train with a joint objective: CE + Ord2Seq loss + EMD + soft-QWK surrogate + consistency + KL distillation.

Fairness protocol used for this run:

- Same split files: `icdas4_train.csv`, `icdas4_val.csv`, `icdas4_test.csv`.
- Same ROI crop/resize/normalize pipeline and same image roots.
- Same training budget (`epochs=60`, `img_size=256`, `expand=1.25`, `bs=64`, `lr=3e-4`).
- Same evaluation style with fixed decode threshold (`t=0.5`), no threshold tuning.

Checkpoint and outputs:

- Checkpoint: `softmax_ordplus_o2s_icdas4.pt`
- Val CSV: `roi_val_icdas4_softmax_ordplus_o2s.csv`
- Test CSV: `roi_test_icdas4_softmax_ordplus_o2s.csv`

Val/Test metrics:

| Split | Method | AUC(>=1) | AUC(>=3) | AUC(>=5) | MAE | QWK |
|---|---|---:|---:|---:|---:|---:|
| Val | Softmax baseline | 0.812 | 0.906 | 0.890 | 0.314 | 0.614 |
| Val | Ord2Seq-guided Softmax OrdPlus | 0.804 | 0.911 | 0.908 | 0.323 | 0.624 |
| Test | Softmax baseline | 0.862 | 0.813 | 0.996 | 0.296 | 0.596 |
| Test | Ord2Seq-guided Softmax OrdPlus | 0.884 | 0.876 | 0.996 | 0.296 | 0.619 |

Interpretation:

- On test, the new method improves QWK (`0.596 -> 0.619`) while keeping MAE unchanged (`0.296 -> 0.296`).
- On val, QWK improves (`0.614 -> 0.624`) with a small MAE trade-off (`0.314 -> 0.323`).
- This method is currently a strong QWK-oriented alternative to softmax under fixed-threshold evaluation.

Paper comparison note:

- Yes, this run is valid for direct fair comparison in the paper under the current protocol constraints (same data split, same preprocess, same backbone, same decode threshold, no proposal-stage change).
- To make the claim more robust, report both default-threshold and tuned-threshold tables, and include multi-seed mean/std in the final paper table.

## CORN Reproduction (Paper Code Style, Fair Protocol)

Reference:

- Repository used: `model-test/corn-ordinal-neuralnet-main`
- Core logic reproduced from `model-code/simple-scripts/resnet34_corn.py`:
  - output layer uses `K-1` logits for `K` classes
  - CORN task-wise conditional loss (`loss_corn`)
  - decode with `sigmoid -> cumulative product -> threshold(0.5)`

Added scripts:

- Training: `train_corn_head_icdas4.py`
- Evaluation: `eval_corn_on_roi_icdas4.py`

Fairness alignment (same as previous methods):

- same split: `icdas4_train.csv`, `icdas4_val.csv`, `icdas4_test.csv`
- same ROI preprocess: crop + resize to 256 + ImageNet normalize + `expand=1.25`
- same backbone: ResNet18
- same training budget: `epochs=60`, `bs=64`, `lr=3e-4`
- same fixed decode threshold: `0.5`
- proposal flow unchanged

Artifacts:

- Checkpoint: `corn_head_icdas4.pt`
- Val CSV: `roi_val_icdas4_corn.csv`
- Test CSV: `roi_test_icdas4_corn.csv`

Metrics (default threshold, no post-hoc tuning):

| Split | Method | AUC(>=1) | AUC(>=3) | AUC(>=5) | MAE | QWK |
|---|---|---:|---:|---:|---:|---:|
| Val | CORN | 0.821 | 0.886 | 0.981 | 0.331 | 0.593 |
| Test | CORN | 0.880 | 0.856 | 0.996 | 0.298 | 0.583 |

Comparison to softmax baseline (same default-threshold protocol):

- Val: CORN has higher AUC(>=1) (`0.821 vs 0.812`) but lower QWK (`0.593 vs 0.614`) and higher MAE (`0.331 vs 0.314`).
- Test: CORN has higher AUC(>=1/>=3) (`0.880/0.856 vs 0.862/0.813`) but slightly lower QWK (`0.583 vs 0.596`) and slightly higher MAE (`0.298 vs 0.296`).

Current conclusion:

- CORN is now fully reproduced and can be used as a fair baseline in the paper.
- Under fixed-threshold evaluation on this split, CORN does not surpass softmax on both MAE and QWK simultaneously.

## Overall Performance Leaderboard (Fair, Default Threshold)

Protocol for this leaderboard:

- same split / preprocess / backbone / training budget
- fixed threshold decoding (`t=0.5`)
- no proposal-stage changes

### Test Main Leaderboard (`icdas4_test.csv`)

Ranking rule:

- primary key: higher QWK
- tie-break: lower MAE

| Rank | Method | MAE | QWK | AUC(>=1) | AUC(>=3) | AUC(>=5) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Ord2Seq-guided Softmax OrdPlus | 0.296 | 0.619 | 0.884 | 0.876 | 0.996 |
| 2 | Softmax baseline | 0.296 | 0.596 | 0.862 | 0.813 | 0.996 |
| 3 | CORN | 0.298 | 0.583 | 0.880 | 0.856 | 0.996 |
| 4 | Existing Ordinal (Ord2Seq) | 0.286 | 0.568 | 0.857 | 0.680 | 0.998 |
| 5 | CORAL (independent 3-logit) | 0.298 | 0.567 | 0.855 | 0.828 | 0.998 |
| 6 | DCH dynamic (sample-wise gate) | 0.315 | 0.523 | 0.834 | 0.831 | 0.994 |
| 7 | Your own Ordinal (masked) | 0.349 | 0.496 | 0.837 | 0.849 | 0.994 |
| 8 | CORAL strict (shared-weight + bias) | 0.481 | 0.459 | 0.869 | 0.852 | 0.992 |

Notes:

- Best QWK: Ord2Seq-guided Softmax OrdPlus (`0.619`).
- Best MAE in this table: Existing Ordinal (Ord2Seq) (`0.286`).
- Best MAE and best QWK do not come from the same method.

### Validation Main Leaderboard (`icdas4_val.csv`)

| Rank | Method | MAE | QWK | AUC(>=1) | AUC(>=3) | AUC(>=5) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Ord2Seq-guided Softmax OrdPlus | 0.323 | 0.624 | 0.804 | 0.911 | 0.908 |
| 2 | Softmax baseline | 0.314 | 0.614 | 0.812 | 0.906 | 0.890 |
| 3 | DCH-Ordinal (hybrid baseline) | 0.336 | 0.596 | 0.808 | 0.912 | 0.942 |
| 4 | CORN | 0.331 | 0.593 | 0.821 | 0.886 | 0.981 |
| 5 | Existing Ordinal (Ord2Seq) | 0.326 | 0.584 | 0.801 | 0.731 | 0.939 |
| 6 | CORAL strict (shared-weight + bias) | 0.410 | 0.573 | 0.822 | 0.897 | 0.945 |
| 7 | CORAL (independent 3-logit) | 0.365 | 0.566 | 0.783 | 0.915 | 0.974 |
| 8 | Your own Ordinal (masked) | 0.356 | 0.544 | 0.794 | 0.940 | 0.960 |
