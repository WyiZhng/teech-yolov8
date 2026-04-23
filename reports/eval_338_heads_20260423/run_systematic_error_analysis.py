#!/usr/bin/env python3
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


CLASS_NAMES = ["0", "A", "B", "C"]
CLASS_TO_NAME = {0: "0", 1: "A", 2: "B", 3: "C"}
NAME_TO_CLASS = {v: k for k, v in CLASS_TO_NAME.items()}

MODELS = [
    "softmax",
    "ord2seq",
    "corn",
    "softmax_ordplus_o2s",
    "softmax_ordplus_o2s_boundary_gpt",
]
SPLITS = ["val", "test"]


@dataclass
class Paths:
    root: Path
    out_dir: Path


def load_prediction_csvs(root: Path) -> Dict[Tuple[str, str], pd.DataFrame]:
    files = {
        ("softmax", "val"): root / "reports/eval_338_heads_20260423/pred_softmax_val_338.csv",
        ("softmax", "test"): root / "reports/eval_338_heads_20260423/pred_softmax_test_338.csv",
        ("ord2seq", "val"): root / "reports/eval_338_heads_20260423/pred_ord2seq_val_338.csv",
        ("ord2seq", "test"): root / "reports/eval_338_heads_20260423/pred_ord2seq_test_338.csv",
        ("corn", "val"): root / "roi_val_icdas4_corn_338.csv",
        ("corn", "test"): root / "roi_test_icdas4_corn_338.csv",
        ("softmax_ordplus_o2s", "val"): root / "roi_val_icdas4_softmax_ordplus_o2s_338.csv",
        ("softmax_ordplus_o2s", "test"): root / "roi_test_icdas4_softmax_ordplus_o2s_338.csv",
        ("softmax_ordplus_o2s_boundary_gpt", "val"): root / "roi_val_icdas4_boundary_gpt_338.csv",
        ("softmax_ordplus_o2s_boundary_gpt", "test"): root / "roi_test_icdas4_boundary_gpt_338.csv",
    }

    out = {}
    for key, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")
        df = pd.read_csv(f)
        out[key] = df
    return out


def normalize_df(df: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    out = df.copy()

    if "gt_class" in out.columns:
        out = out.rename(columns={"gt_class": "ic4"})
    if "pred_class" in out.columns:
        out = out.rename(columns={"pred_class": "pred_ic4"})
    if "y_pred" in out.columns and "pred_ic4" not in out.columns:
        out = out.rename(columns={"y_pred": "pred_ic4"})
    if "y_gt" in out.columns and "ic4" not in out.columns:
        out = out.rename(columns={"y_gt": "ic4"})

    need_cols = ["image_id", "ic4", "pred_ic4"]
    for c in need_cols:
        if c not in out.columns:
            raise ValueError(f"{model}-{split} missing required column: {c}")

    out["ic4"] = out["ic4"].astype(int)
    out["pred_ic4"] = out["pred_ic4"].astype(int)

    if "roi_id" not in out.columns:
        out["roi_id"] = out.groupby("image_id").cumcount()

    out["model"] = model
    out["split"] = split
    return out


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(4), CLASS_NAMES)
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    ax.set_title(title)

    vmax = max(cm.max(), 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            color = "white" if v > vmax * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def error_type_counts(df: pd.DataFrame) -> Dict[str, float]:
    gt = df["ic4"].to_numpy()
    pred = df["pred_ic4"].to_numpy()
    diff = np.abs(gt - pred)

    is_err = gt != pred
    total = len(df)
    total_err = int(is_err.sum())

    pair_01 = ((gt == 0) & (pred == 1)) | ((gt == 1) & (pred == 0))
    pair_12 = ((gt == 1) & (pred == 2)) | ((gt == 2) & (pred == 1))
    pair_23 = ((gt == 2) & (pred == 3)) | ((gt == 3) & (pred == 2))
    cross2 = diff >= 2
    adjacent = diff == 1

    def make(name: str, mask: np.ndarray) -> Dict[str, float]:
        c = int(mask.sum())
        return {
            f"{name}_count": c,
            f"{name}_ratio_all": c / total if total else 0.0,
            f"{name}_ratio_err": c / total_err if total_err else 0.0,
        }

    out = {
        "n_total": total,
        "n_errors": total_err,
        "error_ratio": total_err / total if total else 0.0,
    }
    out.update(make("err_0_A", pair_01))
    out.update(make("err_A_B", pair_12))
    out.update(make("err_B_C", pair_23))
    out.update(make("err_cross_ge2", cross2))
    out.update(make("err_adjacent", adjacent))
    return out


def threshold_to_class_probs(p_ge1: float, p_ge3: float, p_ge5: float) -> np.ndarray:
    p0 = max(0.0, min(1.0, 1.0 - p_ge1))
    p1 = max(0.0, min(1.0, p_ge1 - p_ge3))
    p2 = max(0.0, min(1.0, p_ge3 - p_ge5))
    p3 = max(0.0, min(1.0, p_ge5))
    s = p0 + p1 + p2 + p3
    if s <= 0:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    return np.array([p0, p1, p2, p3], dtype=float) / s


def fmt_probs(arr: np.ndarray) -> str:
    return "0:{:.3f} A:{:.3f} B:{:.3f} C:{:.3f}".format(arr[0], arr[1], arr[2], arr[3])


def fmt_threshold_probs(p1: float, p3: float, p5: float) -> str:
    return "P(>=1):{:.3f} P(>=3):{:.3f} P(>=5):{:.3f}".format(p1, p3, p5)


def dataframe_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join([":---" for _ in cols]) + " |"

    rows = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.{float_digits}f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def safe_crop(img: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    W, H = img.size
    x1 = int(max(0, np.floor(x)))
    y1 = int(max(0, np.floor(y)))
    x2 = int(min(W, np.ceil(x + w)))
    y2 = int(min(H, np.ceil(y + h)))
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img.crop((x1, y1, x2, y2))


def save_case_visual(
    row: pd.Series,
    out_path: Path,
    img_roots: Dict[str, Path],
) -> None:
    split = row["split"]
    image_id = row["image_id"]
    img_path = img_roots[split] / image_id
    if not img_path.exists():
        return

    img = Image.open(img_path).convert("RGB")
    crop = safe_crop(img, row["x"], row["y"], row["w"], row["h"])

    softmax_probs = np.array([row["sm_p0"], row["sm_pA"], row["sm_pB"], row["sm_pC"]], dtype=float)
    ord_probs = threshold_to_class_probs(row["ord_p_ge1"], row["ord_p_ge3"], row["ord_p_ge5"])
    bd_probs = threshold_to_class_probs(row["bd_p_ge1"], row["bd_p_ge3"], row["bd_p_ge5"])

    fig = plt.figure(figsize=(9, 4.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(crop)
    ax1.axis("off")
    ax1.set_title("ROI")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    text = "\n".join([
        f"split: {split}",
        f"image: {image_id}",
        f"roi_id: {int(row['roi_id'])}",
        f"GT: {CLASS_TO_NAME[int(row['ic4'])]}  Pred(boundary_gpt): {CLASS_TO_NAME[int(row['bd_pred'])]}",
        "",
        f"softmax probs: {fmt_probs(softmax_probs)}",
        f"ord2seq threshold probs: {fmt_threshold_probs(row['ord_p_ge1'], row['ord_p_ge3'], row['ord_p_ge5'])}",
        f"ord2seq class probs(derived): {fmt_probs(ord_probs)}",
        f"boundary_gpt threshold probs: {fmt_threshold_probs(row['bd_p_ge1'], row['bd_p_ge3'], row['bd_p_ge5'])}",
        f"boundary_gpt class probs(derived): {fmt_probs(bd_probs)}",
    ])
    ax2.text(0.01, 0.99, text, va="top", ha="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path("/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35")
    out_dir = root / "reports/eval_338_heads_20260423/systematic_error_analysis"
    rep_dir = out_dir / "representative_error_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    img_roots = {
        "val": root / "VOCdevkit_338_matched_userref_balanced/val/images",
        "test": root / "VOCdevkit_338_matched_userref_balanced/test/images",
    }

    raw = load_prediction_csvs(root)
    norm = {(m, s): normalize_df(df, m, s) for (m, s), df in raw.items()}

    # 1) confusion matrices + model-wise error counters
    summary_rows: List[Dict[str, float]] = []
    for model in MODELS:
        for split in SPLITS:
            df = norm[(model, split)]
            cm = compute_confusion(df["ic4"].to_numpy(), df["pred_ic4"].to_numpy(), n_classes=4)
            plot_confusion(
                cm,
                title=f"Confusion Matrix - {model} ({split})",
                out_path=out_dir / f"confusion_matrix_{model}_{split}.png",
            )

            row = {"model": model, "split": split}
            row.update(error_type_counts(df))
            summary_rows.append(row)

    error_summary = pd.DataFrame(summary_rows)
    error_summary.to_csv(out_dir / "top_error_patterns.csv", index=False)

    # 2) boundary_gpt deep-dive merge with softmax/ord2seq
    bd_frames = []
    for split in SPLITS:
        bd = norm[("softmax_ordplus_o2s_boundary_gpt", split)].copy()
        sm = norm[("softmax", split)].copy()
        od = norm[("ord2seq", split)].copy()
        st = norm[("softmax_ordplus_o2s", split)].copy()

        key = ["image_id", "roi_id"]

        bd = bd.rename(columns={"pred_ic4": "bd_pred", "ic4": "ic4"})
        sm = sm.rename(columns={
            "pred_ic4": "sm_pred",
            "p0": "sm_p0",
            "pA": "sm_pA",
            "pB": "sm_pB",
            "pC": "sm_pC",
            "p_ge1": "sm_p_ge1",
        })
        od = od.rename(columns={
            "pred_ic4": "ord_pred",
            "p_ge1": "ord_p_ge1",
            "p_ge3": "ord_p_ge3",
            "p_ge5": "ord_p_ge5",
        })
        st = st.rename(columns={"pred_ic4": "static_pred"})

        keep_sm = key + ["sm_pred", "sm_p0", "sm_pA", "sm_pB", "sm_pC", "sm_p_ge1"]
        keep_od = key + ["ord_pred", "ord_p_ge1", "ord_p_ge3", "ord_p_ge5"]
        keep_st = key + ["static_pred"]

        bd_keep = key + ["split", "ic4", "bd_pred", "p_ge1", "p_ge3", "p_ge5"]
        for c in ["x", "y", "w", "h"]:
            if c in raw[("softmax", split)].columns:
                if c not in bd.columns:
                    bd[c] = raw[("softmax", split)][c].values
                bd_keep.append(c)

        bd = bd[bd_keep].rename(columns={"p_ge1": "bd_p_ge1", "p_ge3": "bd_p_ge3", "p_ge5": "bd_p_ge5"})

        merged = bd.merge(sm[keep_sm], on=key, how="inner").merge(od[keep_od], on=key, how="inner").merge(st[keep_st], on=key, how="inner")

        if len(merged) != len(bd):
            raise RuntimeError(f"Merge size mismatch on {split}: {len(merged)} vs {len(bd)}")

        # sanity check GT alignment
        if (merged["ic4"] < 0).any() or (merged["ic4"] > 3).any():
            raise RuntimeError("GT out of class range")

        bd_frames.append(merged)

    bd_all = pd.concat(bd_frames, ignore_index=True)
    bd_all["is_err"] = bd_all["ic4"] != bd_all["bd_pred"]
    bd_all["abs_diff"] = (bd_all["ic4"] - bd_all["bd_pred"]).abs()
    bd_all["adjacent_err"] = bd_all["is_err"] & (bd_all["abs_diff"] == 1)
    bd_all["cross_ge2_err"] = bd_all["is_err"] & (bd_all["abs_diff"] >= 2)

    # category comparisons
    bd_all["bd_err_sm_ord_correct"] = (
        (bd_all["bd_pred"] != bd_all["ic4"]) & (bd_all["sm_pred"] == bd_all["ic4"]) & (bd_all["ord_pred"] == bd_all["ic4"])
    )
    bd_all["bd_correct_sm_ord_err"] = (
        (bd_all["bd_pred"] == bd_all["ic4"]) & (bd_all["sm_pred"] != bd_all["ic4"]) & (bd_all["ord_pred"] != bd_all["ic4"])
    )
    bd_all["all_three_err"] = (
        (bd_all["bd_pred"] != bd_all["ic4"]) & (bd_all["sm_pred"] != bd_all["ic4"]) & (bd_all["ord_pred"] != bd_all["ic4"])
    )

    # softmax vs static comparison
    bd_all["fixed_vs_static"] = (bd_all["bd_pred"] == bd_all["ic4"]) & (bd_all["static_pred"] != bd_all["ic4"])
    bd_all["new_err_vs_softmax"] = (bd_all["bd_pred"] != bd_all["ic4"]) & (bd_all["sm_pred"] == bd_all["ic4"])

    # top error patterns for boundary_gpt
    err = bd_all[bd_all["is_err"]].copy()
    err["pattern"] = err["ic4"].map(CLASS_TO_NAME) + "->" + err["bd_pred"].map(CLASS_TO_NAME)
    pattern_counts = (
        err.groupby(["split", "pattern", "ic4", "bd_pred"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["split", "count"], ascending=[True, False])
    )

    # Add overall pattern aggregate
    overall = (
        err.groupby(["pattern", "ic4", "bd_pred"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    overall["split"] = "overall"
    pattern_counts_full = pd.concat([pattern_counts, overall], ignore_index=True)

    # top_error_patterns.csv should include both model stats and boundary pattern table for traceability
    with open(out_dir / "top_error_patterns.csv", "a", encoding="utf-8") as f:
        f.write("\n# boundary_gpt_error_patterns\n")
    pattern_counts_full.to_csv(out_dir / "top_error_patterns_boundary_gpt.csv", index=False)

    # choose 3 most common overall patterns and representative sample per pattern
    top3 = overall.head(3).copy()
    reps = []
    for _, r in top3.iterrows():
        g, p = int(r["ic4"]), int(r["bd_pred"])
        sub = err[(err["ic4"] == g) & (err["bd_pred"] == p)].copy()
        # confidence of boundary predicted class via derived class probs
        def pred_conf(row: pd.Series) -> float:
            probs = threshold_to_class_probs(row["bd_p_ge1"], row["bd_p_ge3"], row["bd_p_ge5"])
            return float(probs[int(row["bd_pred"])])

        sub["bd_pred_conf"] = sub.apply(pred_conf, axis=1)
        rep = sub.sort_values("bd_pred_conf", ascending=False).iloc[0]
        reps.append(rep)

        fn = f"pattern_{CLASS_TO_NAME[g]}_to_{CLASS_TO_NAME[p]}__{rep['split']}__{Path(rep['image_id']).stem}_roi{int(rep['roi_id'])}.png"
        save_case_visual(rep, rep_dir / fn, img_roots)

    rep_df = pd.DataFrame(reps)
    if not rep_df.empty:
        rep_df["pattern"] = rep_df["ic4"].map(CLASS_TO_NAME) + "->" + rep_df["bd_pred"].map(CLASS_TO_NAME)
        rep_df[["split", "image_id", "roi_id", "ic4", "bd_pred", "pattern"]].to_csv(out_dir / "representative_error_cases_index.csv", index=False)

    # 3) markdown reports
    # model-wise error summary markdown
    lines = []
    lines.append("# ICDAS4 ROI 错例分析总结（val/test）")
    lines.append("")
    lines.append("## 1. 各模型错误类型统计")
    lines.append("")
    for split in SPLITS:
        sub = error_summary[error_summary["split"] == split].copy()
        disp = sub[[
            "model", "n_total", "n_errors", "error_ratio",
            "err_0_A_count", "err_0_A_ratio_err",
            "err_A_B_count", "err_A_B_ratio_err",
            "err_B_C_count", "err_B_C_ratio_err",
            "err_cross_ge2_count", "err_cross_ge2_ratio_err",
            "err_adjacent_count", "err_adjacent_ratio_err",
        ]].copy()
        lines.append(f"### {split.upper()} 集")
        lines.append("")
        lines.append(dataframe_to_markdown(disp, float_digits=4))
        lines.append("")

    # boundary deep-dive
    lines.append("## 2. boundary_gpt 重点分析")
    lines.append("")
    for split in SPLITS:
        sub = bd_all[bd_all["split"] == split]
        n = len(sub)
        e = int((sub["bd_pred"] != sub["ic4"]).sum())
        adj = int(sub["adjacent_err"].sum())
        cross = int(sub["cross_ge2_err"].sum())
        lines.append(f"### {split.upper()} 集")
        lines.append("")
        lines.append(f"- boundary_gpt 总样本: {n}, 错误: {e}, 错误率: {e/n:.4f}")
        lines.append(f"- 相邻错级: {adj} ({adj/e:.4f} of errors)")
        lines.append(f"- 跨两级及以上: {cross} ({cross/e:.4f} of errors)")

        # category stats
        c1 = int(sub["bd_err_sm_ord_correct"].sum())
        c2 = int(sub["bd_correct_sm_ord_err"].sum())
        c3 = int(sub["all_three_err"].sum())
        lines.append(f"- boundary_gpt 错, softmax+ord2seq 都对: {c1}")
        lines.append(f"- boundary_gpt 对, softmax+ord2seq 都错: {c2}")
        lines.append(f"- softmax+ord2seq+boundary_gpt 三者都错: {c3}")
        lines.append("")

        # top pattern
        sp = pattern_counts[pattern_counts["split"] == split].head(8)
        if not sp.empty:
            lines.append("最常见误判模式（boundary_gpt）:")
            lines.append("")
            lines.append(dataframe_to_markdown(sp[["pattern", "count"]], float_digits=4))
            lines.append("")

    # extra requirements
    lines.append("## 3. 额外统计")
    lines.append("")
    bd_err = bd_all[bd_all["is_err"]]
    n_err = len(bd_err)
    a0 = int((((bd_err["ic4"] == 0) & (bd_err["bd_pred"] == 1)) | ((bd_err["ic4"] == 1) & (bd_err["bd_pred"] == 0))).sum())
    lines.append(f"- A↔0 是否最大误判来源: {a0}/{n_err} ({(a0/n_err if n_err else 0):.4f})")

    fixed = bd_all[bd_all["fixed_vs_static"]].copy()
    if len(fixed):
        fixed["pair"] = fixed["ic4"].map(CLASS_TO_NAME) + "<-" + fixed["static_pred"].map(CLASS_TO_NAME)
        top_fix = fixed.groupby("pair", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        lines.append("- 相比 static ordplus，boundary_gpt 修正成功最多类别:")
        lines.append(dataframe_to_markdown(top_fix.head(5), float_digits=4))

    newe = bd_all[bd_all["new_err_vs_softmax"]].copy()
    if len(newe):
        newe["pair"] = newe["ic4"].map(CLASS_TO_NAME) + "->" + newe["bd_pred"].map(CLASS_TO_NAME)
        top_new = newe.groupby("pair", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        lines.append("- 相比 softmax，boundary_gpt 新增错误最多类别:")
        lines.append(dataframe_to_markdown(top_new.head(5), float_digits=4))

    # evidence-based cause hints
    lines.append("")
    lines.append("## 4. 证据驱动原因判断")
    lines.append("")

    cross_rate = float((bd_err["abs_diff"] >= 2).mean()) if len(bd_err) else 0.0
    adj_rate = float((bd_err["abs_diff"] == 1).mean()) if len(bd_err) else 0.0
    long_tail_ratio = float((bd_err["ic4"] >= 2).mean()) if len(bd_err) else 0.0

    lines.append(f"- 相邻错级占比: {adj_rate:.4f}；跨两级占比: {cross_rate:.4f}。")
    lines.append(f"- 错误中 GT 属于 B/C 的占比(长尾近似): {long_tail_ratio:.4f}。")
    lines.append("- 若相邻错级远高于跨级错，通常更支持边界模糊/标注近边界样本导致；若跨级占比较高，常见于 ROI 裁剪或严重语义缺失。")

    (out_dir / "error_pattern_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # dedicated boundary-vs-softmax-vs-ord2seq report with case lists
    cmp_lines = []
    cmp_lines.append("# boundary_gpt vs softmax vs ord2seq 对比报告")
    cmp_lines.append("")
    for split in SPLITS:
        sub = bd_all[bd_all["split"] == split].copy()
        cmp_lines.append(f"## {split.upper()} 集")
        cmp_lines.append("")

        def list_cases(mask_col: str, title: str, max_rows: int = 40):
            m = sub[sub[mask_col]].copy()
            cmp_lines.append(f"### {title} (n={len(m)})")
            cmp_lines.append("")
            if len(m) == 0:
                cmp_lines.append("无。")
                cmp_lines.append("")
                return
            show = m[["image_id", "roi_id", "ic4", "sm_pred", "ord_pred", "bd_pred"]].head(max_rows).copy()
            show["ic4"] = show["ic4"].map(CLASS_TO_NAME)
            show["sm_pred"] = show["sm_pred"].map(CLASS_TO_NAME)
            show["ord_pred"] = show["ord_pred"].map(CLASS_TO_NAME)
            show["bd_pred"] = show["bd_pred"].map(CLASS_TO_NAME)
            cmp_lines.append(dataframe_to_markdown(show, float_digits=4))
            cmp_lines.append("")

        list_cases("adjacent_err", "boundary_gpt 相邻错级")
        list_cases("cross_ge2_err", "boundary_gpt 跨两级及以上错误")
        list_cases("bd_err_sm_ord_correct", "boundary_gpt 错, softmax+ord2seq 都对")
        list_cases("bd_correct_sm_ord_err", "boundary_gpt 对, softmax+ord2seq 都错")
        list_cases("all_three_err", "softmax+ord2seq+boundary_gpt 三者都错")

    (out_dir / "boundary_vs_softmax_vs_ord2seq_comparison.md").write_text("\n".join(cmp_lines), encoding="utf-8")

    print(f"Analysis completed. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
