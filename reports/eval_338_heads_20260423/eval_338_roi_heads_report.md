# 338数据集 ROI 序位头评估报告（Val/Test）

## 1) 训练状态确认

已确认在338新划分ROI上完成训练并产出权重：

- `softmax_head_icdas4_338.pt`（43M，时间：4月22日 17:05）
- `ord2seq_head_icdas4_338.pt`（52M，时间：4月22日 17:28）

## 2) 评估设置

- Python环境：`/data2/home/HZNU_ZWY/anaconda3/envs/yolo/bin/python`
- Val ROI：`derived_338_matched_userref_balanced/icdas4_val.csv`（42图，663 ROI）
- Test ROI：`derived_338_matched_userref_balanced/icdas4_test.csv`（43图，674 ROI）
- 评估脚本：
  - Softmax：`tools/eval_softmax_on_roi_icdas4.py`
  - Ord2Seq：`tools/eval_ordinal_on_roi_icdas4.py`

## 3) 核心指标汇总

| Model | Split | n_images | n_rois | AUC>=1 | AUC>=3 | AUC>=5 | MAE | QWK | Top-1 | Top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| softmax | val | 42 | 663 | 0.832 | 0.914 | 0.976 | 0.338 | **0.720** | 0.881 (37/42) | 0.976 (41/42) |
| softmax | test | 43 | 674 | **0.879** | **0.883** | **0.931** | **0.306** | **0.735** | **0.878 (36/41)** | **0.951 (39/41)** |
| ord2seq | val | 42 | 663 | **0.851** | 0.809 | **0.978** | **0.320** | 0.718 | **0.905 (38/42)** | 0.976 (41/42) |
| ord2seq | test | 43 | 674 | 0.848 | 0.822 | 0.906 | 0.346 | 0.664 | 0.837 (36/43) | 0.884 (38/43) |

## 4) 简要结论

- 训练已完成，两个头都可正常评估并输出预测CSV。
- 在**Test**上，`softmax_head_icdas4_338.pt`当前整体表现更稳（更低MAE、更高QWK）。
- 在**Val**上，ord2seq在`AUC>=1`和`AUC>=5`略有优势，但`AUC>=3`与QWK弱于softmax，提示中等级别区分仍有改进空间。

## 5) 结果文件

- 指标汇总CSV：`reports/eval_338_heads_20260423/summary_metrics.csv`
- 报告MD：`reports/eval_338_heads_20260423/eval_338_roi_heads_report.md`
- 逐ROI预测：
  - `reports/eval_338_heads_20260423/pred_softmax_val_338.csv`
  - `reports/eval_338_heads_20260423/pred_softmax_test_338.csv`
  - `reports/eval_338_heads_20260423/pred_ord2seq_val_338.csv`
  - `reports/eval_338_heads_20260423/pred_ord2seq_test_338.csv`
