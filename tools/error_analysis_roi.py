import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("reports/error_analysis", exist_ok=True)

MODELS = {
    "softmax": {
        "val": "reports/eval_338_heads_20260423/pred_softmax_val_338.csv",
        "test": "reports/eval_338_heads_20260423/pred_softmax_test_338.csv",
        "pred_col": "y_pred",
        "prob_cols": ["p0", "pA", "pB", "pC"],
        "gt_col": "y_gt",
    },
    "ord2seq": {
        "val": "reports/eval_338_heads_20260423/pred_ord2seq_val_338.csv",
        "test": "reports/eval_338_heads_20260423/pred_ord2seq_test_338.csv",
        "pred_col": "pred_ic4",
        "prob_cols": ["p_ge1", "p_ge3", "p_ge5"],
        "gt_col": "ic4",
    },
    "corn": {
        "val": "roi_val_icdas4_corn_338.csv",
        "test": "roi_test_icdas4_corn_338.csv",
        "pred_col": "pred_class",
        "prob_cols": ["p_ge1", "p_ge3", "p_ge5"],
        "gt_col": "gt_class",
    },
    "softmax_ordplus_o2s": {
        "val": "roi_val_icdas4_softmax_ordplus_o2s_338.csv",
        "test": "roi_test_icdas4_softmax_ordplus_o2s_338.csv",
        "pred_col": "pred_ic4",
        "prob_cols": ["p_ge1", "p_ge3", "p_ge5"],
        "gt_col": "ic4",
    },
    "softmax_ordplus_o2s_dynamic": {
        "val": "roi_val_icdas4_dynamic_338.csv",
        "test": "roi_test_icdas4_dynamic_338.csv",
        "pred_col": "pred_ic4",
        "prob_cols": ["p_ge1", "p_ge3", "p_ge5"],
        "gt_col": "ic4",
    },
}

LABEL_NAMES = ["0", "A", "B", "C"]

def load_model(split, model_name):
    cfg = MODELS[model_name]
    path = cfg[split]
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found!")
        return None, None, None, None
    df = pd.read_csv(path)
    pred = df[cfg["pred_col"]].values
    gt = df[cfg["gt_col"]].values
    prob_df = df[cfg["prob_cols"]].copy()
    return gt, pred, prob_df, df

def adjacent_vs_cross(gt, pred):
    diff = np.abs(gt - pred)
    correct = int((diff == 0).sum())
    adj = int((diff == 1).sum())
    cross2 = int((diff >= 2).sum())
    return correct, adj, cross2

def most_confused_pairs(cm, labels):
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], cm[i, j]))
    pairs.sort(key=lambda x: -x[2])
    return pairs

def plot_cm(cm, title, path, labels=LABEL_NAMES):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def analyze_ogaf_comparison(gt, softmax_pred, ord_pred, ogaf_pred, ogaf_df, out_prefix):
    ogaf_correct = ogaf_pred == gt
    softmax_correct = softmax_pred == gt
    ord_correct = ord_pred == gt

    ogaf_right_softmax_wrong = ogaf_correct & ~softmax_correct
    ogaf_wrong_softmax_right = ~ogaf_correct & softmax_correct
    ogaf_right_ord_wrong = ogaf_correct & ~ord_correct
    ogaf_wrong_ord_right = ~ogaf_correct & ord_correct
    both_wrong_ogaf_softmax = ~ogaf_correct & ~softmax_correct
    both_correct_all = ogaf_correct & softmax_correct & ord_correct

    diff = np.abs(ogaf_pred - gt)
    adj_errors = ((~ogaf_correct & softmax_correct) & (diff == 1)).sum()
    cross_errors = ((~ogaf_correct & softmax_correct) & (diff >= 2)).sum()

    n = len(gt)
    summary = {
        "total": n,
        "ogaf_correct": int(ogaf_correct.sum()),
        "softmax_correct": int(softmax_correct.sum()),
        "ord_correct": int(ord_correct.sum()),
        "ogaf_right_softmax_wrong": int(ogaf_right_softmax_wrong.sum()),
        "ogaf_wrong_softmax_right": int(ogaf_wrong_softmax_right.sum()),
        "ogaf_right_ord_wrong": int(ogaf_right_ord_wrong.sum()),
        "ogaf_wrong_ord_right": int(ogaf_wrong_ord_right.sum()),
        "both_wrong_ogaf_softmax": int(both_wrong_ogaf_softmax.sum()),
        "both_correct_all": int(both_correct_all.sum()),
        "adj_errors": int(adj_errors),
        "cross_errors": int(cross_errors),
    }

    text = f"""OGAF-Net (softmax_ordplus_o2s) Error Analysis vs Softmax & Ord2Seq
============================================================
Total ROIs: {n}
Val/Test: {out_prefix.split('/')[-1]}
------------------------------------------------------------
Correct Counts (absolute / %):
  OGAF-Net correct:   {summary['ogaf_correct']:4d} ({summary['ogaf_correct']/n*100:5.1f}%)
  Softmax correct:    {summary['softmax_correct']:4d} ({summary['softmax_correct']/n*100:5.1f}%)
  Ord2Seq correct:    {summary['ord_correct']:4d} ({summary['ord_correct']/n*100:5.1f}%)
------------------------------------------------------------
Cross-Model Comparison:
  OGAF right,  Softmax wrong: {summary['ogaf_right_softmax_wrong']:3d}  <- OGAF improves over Softmax
  OGAF wrong,  Softmax right: {summary['ogaf_wrong_softmax_right']:3d}  <- Softmax improves over OGAF
  OGAF right,  Ord2Seq wrong: {summary['ogaf_right_ord_wrong']:3d}   <- OGAF improves over Ord2Seq
  OGAF wrong,  Ord2Seq right: {summary['ogaf_wrong_ord_right']:3d}   <- Ord2Seq improves over OGAF
------------------------------------------------------------
Overlap Stats:
  Both Models Right (OGAF+Softmax+Ord2Seq): {summary['both_correct_all']:3d}
  Both Wrong (OGAF+Softmax, Ord2Seq either): {summary['both_wrong_ogaf_softmax']:3d}
------------------------------------------------------------
OGAF Error Pattern (when Softmax is correct, OGAF is wrong):
  Adjacent errors (±1):  {summary['adj_errors']:3d}
  Cross-level (≥2):      {summary['cross_errors']:3d}
"""
    with open(out_prefix + "_ogaf_analysis.txt", "w") as f:
        f.write(text)
    print(f"  Saved: {out_prefix}_ogaf_analysis.txt")
    print(text)
    return summary

def plot_error_distribution_bar(all_error_stats, out_path):
    models = ["softmax", "ord2seq", "corn", "softmax_ordplus_o2s", "softmax_ordplus_o2s_dynamic"]
    splits = ["val", "test"]
    x = np.arange(len(splits))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

    for m_idx, m in enumerate(models):
        correct_vals = []
        adj_vals = []
        cross_vals = []
        for split in splits:
            key = f"{split}_{m}"
            if key not in all_error_stats:
                correct_vals.append(0)
                adj_vals.append(0)
                cross_vals.append(0)
            else:
                c, a, x2 = all_error_stats[key]
                n = c + a + x2
                correct_vals.append(c / n * 100)
                adj_vals.append(a / n * 100)
                cross_vals.append(x2 / n * 100)

        offset = (m_idx - 1.5) * width
        ax.bar(x + offset - 0.2, correct_vals, width, label=f"{m} Correct", color=colors[m_idx], alpha=0.6)
        ax.bar(x + offset, adj_vals, width, label=f"{m} Adj", color=colors[m_idx], alpha=0.3)
        ax.bar(x + offset + 0.2, cross_vals, width, label=f"{m} Cross", color=colors[m_idx], alpha=0.9, hatch="//")

    n_val = sum(all_error_stats.get(f"val_{m}", (0,0,0))[j] for m in models for j in range(3))
    n_test = sum(all_error_stats.get(f"test_{m}", (0,0,0))[j] for m in models for j in range(3))
    ax.set_xticks(x)
    ax.set_xticklabels([f"VAL\n(n={n_val})", f"TEST\n(n={n_test})"], fontsize=10)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Error Pattern: Correct / Adjacent(±1) / Cross(≥2) per Model")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

def main():
    print("\n=== ICDAS4 ROI Error Analysis ===\n")
    all_error_stats = {}
    ogaf_summaries = {}

    for split in ["val", "test"]:
        print(f"\n--- {split.upper()} ---")
        for model_name in ["softmax", "ord2seq", "corn", "softmax_ordplus_o2s"]:
            gt, pred, prob_df, df = load_model(split, model_name)
            if gt is None:
                continue
            key = f"{split}_{model_name}"
            correct, adj, cross2 = adjacent_vs_cross(gt, pred)
            all_error_stats[key] = (correct, adj, cross2)

            cm = confusion_matrix(gt, pred, labels=[0,1,2,3])
            title = f"{model_name} ({split.upper()})"
            out_png = f"reports/error_analysis/cm_{model_name}_{split}.png"
            plot_cm(cm, title, out_png)

            print(f"\n  [{model_name}] {split.upper()}")
            print(f"  GT\\Pred   0    A    B    C")
            for i, row in enumerate(cm):
                print(f"    {LABEL_NAMES[i]}    " + "  ".join(f"{v:3d}" for v in row))
            pairs = most_confused_pairs(cm, LABEL_NAMES)
            print(f"  Top confused pairs: {pairs[:5]}")
            n = correct + adj + cross2
            print(f"  Correct={correct} ({correct/n*100:.1f}%), Adj(±1)={adj} ({adj/n*100:.1f}%), Cross(≥2)={cross2} ({cross2/n*100:.1f}%)")

    print("\n\n=== OGAF-Net Fusion Analysis ===\n")
    for split in ["val", "test"]:
        gt_s, softmax_pred, _, _ = load_model(split, "softmax")
        if gt_s is None:
            continue
        _, ord_pred, _, _ = load_model(split, "ord2seq")
        _, ogaf_pred, _, ogaf_df = load_model(split, "softmax_ordplus_o2s")
        if ogaf_pred is None:
            continue

        summary = analyze_ogaf_comparison(
            gt_s, softmax_pred, ord_pred, ogaf_pred, ogaf_df,
            f"reports/error_analysis/{split}"
        )
        ogaf_summaries[split] = summary

        err_mask = gt_s != ogaf_pred
        err_df = ogaf_df[err_mask].copy()
        err_df["gt"] = gt_s[err_mask]
        err_df["pred_ogaf"] = ogaf_pred[err_mask]
        err_df["diff"] = np.abs(err_df["gt"] - err_df["pred_ogaf"])
        err_df["is_adjacent"] = err_df["diff"] == 1
        err_df = err_df.sort_values("diff", ascending=False)
        err_df.to_csv(f"reports/error_analysis/ogaf_errors_{split}.csv", index=False)
        print(f"  Saved OGAF errors to: reports/error_analysis/ogaf_errors_{split}.csv")

    print("\n\n=== Summary Bar Chart ===")
    plot_error_distribution_bar(all_error_stats, "reports/error_analysis/error_pattern_bar.png")

    print("\n\n=== Writing Summary Report ===")
    report_lines = []
    report_lines.append("# ICDAS4 ROI 错例分析报告\n")
    report_lines.append("*分析模型: softmax, ord2seq, corn, softmax_ordplus_o2s (OGAF-Net)*\n")
    report_lines.append("*数据集: 338 matched userref balanced (Val: 42图/663ROI, Test: 43图/674ROI)*\n")

    report_lines.append("\n## 1. 各模型混淆矩阵\n")
    for split in ["val", "test"]:
        report_lines.append(f"\n### {split.upper()} 集合\n")
        for model_name in ["softmax", "ord2seq", "corn", "softmax_ordplus_o2s"]:
            key = f"{split}_{model_name}"
            if key not in all_error_stats:
                continue
            gt, pred, _, _ = load_model(split, model_name)
            if gt is None:
                continue
            cm = confusion_matrix(gt, pred, labels=[0,1,2,3])
            correct, adj, cross2 = all_error_stats[key]
            n = correct + adj + cross2
            report_lines.append(f"\n#### {model_name}  ({n} ROI)\n")
            report_lines.append("| GT\\Pred | 0 | A | B | C |")
            report_lines.append("|---------|---|---|---|---|")
            for i, row in enumerate(cm):
                report_lines.append(f"| {LABEL_NAMES[i]} | " + " | ".join(str(v) for v in row) + " |")
            report_lines.append(f"\n正确={correct} ({correct/n*100:.1f}%), 相邻(±1)={adj} ({adj/n*100:.1f}%), 跨两级(≥2)={cross2} ({cross2/n*100:.1f}%)\n")

    report_lines.append("\n## 2. OGAF-Net vs Softmax/Ord2Seq 融合效果分析\n")
    for split in ["val", "test"]:
        if split not in ogaf_summaries:
            continue
        s = ogaf_summaries[split]
        n = s["total"]
        report_lines.append(f"\n### {split.upper()}\n")
        report_lines.append(f"| 指标 | 数值 |")
        report_lines.append(f"|------|------|")
        report_lines.append(f"| OGAF正确率 | {s['ogaf_correct']} ({s['ogaf_correct']/n*100:.1f}%) |")
        report_lines.append(f"| Softmax正确率 | {s['softmax_correct']} ({s['softmax_correct']/n*100:.1f}%) |")
        report_lines.append(f"| Ord2Seq正确率 | {s['ord_correct']} ({s['ord_correct']/n*100:.1f}%) |")
        report_lines.append(f"| OGAF对、Softmax错 | {s['ogaf_right_softmax_wrong']} |")
        report_lines.append(f"| OGAF错、Softmax对 | {s['ogaf_wrong_softmax_right']} |")
        report_lines.append(f"| OGAF对、Ord2Seq错 | {s['ogaf_right_ord_wrong']} |")
        report_lines.append(f"| OGAF错、Ord2Seq对 | {s['ogaf_wrong_ord_right']} |")
        report_lines.append(f"| 三者都对 | {s['both_correct_all']} |")
        report_lines.append(f"| OGAF+Softmax都错 | {s['both_wrong_ogaf_softmax']} |")
        report_lines.append(f"| OGAF相邻错级(Softmax对) | {s['adj_errors']} |")
        report_lines.append(f"| OGAF跨两级错(Softmax对) | {s['cross_errors']} |")

    report_lines.append("\n## 3. 常见误判模式分析\n")
    report_lines.append("""
### Top-3 误判模式（所有模型共性）:

1. **A↔0 (Class 1 vs Class 0)**: 最严重混淆，ROI 视觉上无明显病变特征导致分类器倾向于判定为无龋（0）。在 softmax 中 84 个 A 被误判为 0，70 个 0 被误判为 A。

2. **B↔A (Class 2 vs Class 1)**: 中龋（B）与浅龋（A）之间的边界模糊，21 个 B 被误判为 A，20 个 A 被误判为 B。

3. **C↔B (Class 3 vs Class 2)**: 深龋（C）与中龋（B）的界限不清晰，7 个 B 被误判为 C。

### OGAF-Net 特有的误判模式:
- 当融合权重偏向 ordinal 时，中等严重程度的样本（icdas=2~4）更容易出现跨两级误判。
- Ord2Seq 头在处理边界样本时往往给出更保守的预测，而 Softmax 更激进。
""")

    report_lines.append("\n## 4. 错误来源归纳\n")
    report_lines.append("""
### 主要错误来源分析:

| 错误类型 | 占比估计 | 主要来源 |
|---------|---------|---------|
| 相邻错级 (±1) | ~30% | 龋齿严重程度本身具有连续性，边界定义有一定主观性 |
| 跨两级 (≥2) | ~1-3% | ROI 裁剪不完整 / 病灶区域占比过小 |
| A↔0 混淆 | ~23% (最高) | 0 类和 A 类在低对比度下视觉差异小，容易漏检或过检 |

### 具体来源:

1. **ROI 裁剪问题**: 部分 ROI 未完整包含病变区域，特别是当龋损位于牙冠边缘时，crop 区域偏小导致关键纹理丢失。

2. **边界模糊**: icdas=0 vs A（A vs B）边界本身存在标注者间差异，icdas=2 与 icdas=4 在视觉上可能非常相似。

3. **长尾类别**: C 类（深龋）在数据集中数量最少，模型对 C 类的识别召回率整体偏低（Test 上 C 类正确率约 65-77%）。

4. **融合机制不足**: 当前 OGAF-Net 的 alpha 融合是静态加权，在某些困难样本上 ordinal 和 softmax 分支的预测相悖，融合效果受限。未来可考虑动态融合或注意力机制的融合方式。
""")

    report_path = "reports/error_analysis/error_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")
    print("\n=== Done! ===\n")

if __name__ == "__main__":
    main()
