# CLOC / ORDER Integration Summary

## Scope

This integration adds two new ROI baselines to the fixed ICDAS4 two-stage pipeline:

- `order`: CE + ORDER ordinal regularization
- `cloc`: CE + CLOC ordinal contrastive regularization

The following were intentionally kept fixed for fairness:

- same split
- same ROI crop / preprocess
- same ResNet18 backbone
- same proposal stage
- same default softmax decoding and ROI evaluation protocol
- same default batch size / epoch / lr as existing ResNet18 ROI baselines

## What Was Reused

### ORDER

- Reused from official repository:
  - the core idea of weighted pairwise ordinal distance regularization
  - the default coefficient style (`ld` / `order_weight`)
- Rewritten for this ROI pipeline:
  - the loss implementation was rewritten equivalently with `torch.cdist(..., p=1)` instead of `torchmetrics.functional.pairwise_manhattan_distance`
  - the official training/data code was not reused because it assumes a different dataset format and model API

### CLOC

- Reused from official repository:
  - the official `OrdinalContrastiveLoss_mm` / `OrdinalContrastiveLoss_sm` implementation is loaded directly from `model-test/CLOC-main/loss.py`
  - the official two-phase optimization logic is preserved in spirit: phase one learns model + margins, phase two fixes margins and updates model only
- Adapted for this ROI pipeline:
  - the backbone is fixed to ResNet18
  - the head is fixed to a standard 4-class linear classifier
  - the CLOC loss is applied on the 4-class logits for minimal-invasive integration into the existing softmax-style pipeline

## Fairness Notes

- `ORDER` is primarily a loss-level method. In this integration it does not change the head design; it only regularizes the ResNet18 feature space while the classifier remains a standard 4-class softmax head.
- `CLOC` is primarily a loss-level / representation-level method. In this integration it also keeps the same simple 4-class head and adds the official ordinal contrastive loss on top.
- Because both methods share the same ROI crop, split, backbone, logits decoding, and evaluation outputs, they are fairer comparisons to your existing softmax-family baselines than using the original repositories end-to-end.

## Training / Eval Entry Points

- `train_order_head_icdas4.py`
- `tools/eval_order_on_roi_icdas4.py`
- `train_cloc_head_icdas4.py`
- `tools/eval_cloc_on_roi_icdas4.py`

## Suggested Commands

```bash
python train_order_head_icdas4.py --train_csv icdas4_train.csv --val_csv icdas4_val.csv --img_root_train <train_root> --img_root_val <val_root> --out ckpts/order_head_icdas4.pt
python tools/eval_order_on_roi_icdas4.py --roi_csv icdas4_test.csv --img_root <test_root> --ckpt ckpts/order_head_icdas4.pt --out_csv pred_csv/roi_test_icdas4_order.csv

python train_cloc_head_icdas4.py --train_csv icdas4_train.csv --val_csv icdas4_val.csv --img_root_train <train_root> --img_root_val <val_root> --out ckpts/cloc_head_icdas4.pt
python tools/eval_cloc_on_roi_icdas4.py --roi_csv icdas4_test.csv --img_root <test_root> --ckpt ckpts/cloc_head_icdas4.pt --out_csv pred_csv/roi_test_icdas4_cloc.csv
```

## Result Table

| method | split | AUC>=1 | AUC>=3 | AUC>=5 | MAE | QWK | Top-1 | Top-3 | ckpt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| order | val | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| order | test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| cloc | val | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| cloc | test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
