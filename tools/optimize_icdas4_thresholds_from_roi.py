import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, roc_auc_score


def extract_gt_and_probs(df: pd.DataFrame):
    if "gt_class" in df.columns:
        y = df["gt_class"].astype(int).to_numpy()
    elif "ic4" in df.columns:
        y = df["ic4"].astype(int).to_numpy()
    else:
        raise ValueError("CSV must contain gt_class or ic4")

    for c in ["p_ge1", "p_ge3", "p_ge5"]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    p1 = df["p_ge1"].astype(float).to_numpy()
    p3 = df["p_ge3"].astype(float).to_numpy()
    p5 = df["p_ge5"].astype(float).to_numpy()
    return y, p1, p3, p5


def decode_with_thresholds(p1, p3, p5, t1, t3, t5):
    pred = np.zeros_like(p1, dtype=np.int64)
    pred[p1 >= t1] = 1
    pred[p3 >= t3] = 2
    pred[p5 >= t5] = 3
    return pred


def safe_auc(y_bin, s):
    y_bin = np.asarray(y_bin)
    if np.unique(y_bin).size < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, s))


def evaluate(y, p1, p3, p5, t1, t3, t5):
    pred = decode_with_thresholds(p1, p3, p5, t1, t3, t5)
    qwk = float(cohen_kappa_score(y, pred, weights="quadratic"))
    mae = float(mean_absolute_error(y, pred))
    auc1 = safe_auc((y >= 1).astype(int), p1)
    auc3 = safe_auc((y >= 2).astype(int), p3)
    auc5 = safe_auc((y >= 3).astype(int), p5)
    return {
        "qwk": qwk,
        "mae": mae,
        "auc_ge1": auc1,
        "auc_ge3": auc3,
        "auc_ge5": auc5,
        "pred": pred,
    }


def grid_search(y, p1, p3, p5, start=0.1, end=0.9, step=0.02, objective="qwk"):
    grid = np.arange(start, end + 1e-9, step)

    best = None
    for t1 in grid:
        for t3 in grid:
            for t5 in grid:
                out = evaluate(y, p1, p3, p5, t1, t3, t5)
                cand = {
                    "t1": float(t1),
                    "t3": float(t3),
                    "t5": float(t5),
                    "qwk": out["qwk"],
                    "mae": out["mae"],
                }
                if best is None:
                    best = cand
                    continue
                if objective == "mae":
                    # primary lower MAE, tie-break higher QWK
                    better = (cand["mae"] < best["mae"]) or (
                        abs(cand["mae"] - best["mae"]) < 1e-12 and cand["qwk"] > best["qwk"]
                    )
                else:
                    # primary higher QWK, tie-break lower MAE
                    better = (cand["qwk"] > best["qwk"]) or (
                        abs(cand["qwk"] - best["qwk"]) < 1e-12 and cand["mae"] < best["mae"]
                    )
                if better:
                    best = cand
    return best


def main(a):
    val_df = pd.read_csv(a.val_csv)
    test_df = pd.read_csv(a.test_csv)

    yv, p1v, p3v, p5v = extract_gt_and_probs(val_df)
    yt, p1t, p3t, p5t = extract_gt_and_probs(test_df)

    base_val = evaluate(yv, p1v, p3v, p5v, 0.5, 0.5, 0.5)
    base_test = evaluate(yt, p1t, p3t, p5t, 0.5, 0.5, 0.5)

    best = grid_search(
        yv,
        p1v,
        p3v,
        p5v,
        start=a.start,
        end=a.end,
        step=a.step,
        objective=a.objective,
    )

    tuned_val = evaluate(yv, p1v, p3v, p5v, best["t1"], best["t3"], best["t5"])
    tuned_test = evaluate(yt, p1t, p3t, p5t, best["t1"], best["t3"], best["t5"])

    val_out = val_df.copy()
    test_out = test_df.copy()
    val_out["pred_class_tuned"] = tuned_val["pred"]
    test_out["pred_class_tuned"] = tuned_test["pred"]
    val_out.to_csv(a.out_val_csv, index=False)
    test_out.to_csv(a.out_test_csv, index=False)

    summary = {
        "name": a.name,
        "objective": a.objective,
        "best_thresholds": {"t1": best["t1"], "t3": best["t3"], "t5": best["t5"]},
        "base_val": {
            "qwk": base_val["qwk"],
            "mae": base_val["mae"],
            "auc_ge1": base_val["auc_ge1"],
            "auc_ge3": base_val["auc_ge3"],
            "auc_ge5": base_val["auc_ge5"],
        },
        "tuned_val": {
            "qwk": tuned_val["qwk"],
            "mae": tuned_val["mae"],
            "auc_ge1": tuned_val["auc_ge1"],
            "auc_ge3": tuned_val["auc_ge3"],
            "auc_ge5": tuned_val["auc_ge5"],
        },
        "base_test": {
            "qwk": base_test["qwk"],
            "mae": base_test["mae"],
            "auc_ge1": base_test["auc_ge1"],
            "auc_ge3": base_test["auc_ge3"],
            "auc_ge5": base_test["auc_ge5"],
        },
        "tuned_test": {
            "qwk": tuned_test["qwk"],
            "mae": tuned_test["mae"],
            "auc_ge1": tuned_test["auc_ge1"],
            "auc_ge3": tuned_test["auc_ge3"],
            "auc_ge5": tuned_test["auc_ge5"],
        },
        "out_val_csv": a.out_val_csv,
        "out_test_csv": a.out_test_csv,
    }

    os.makedirs(os.path.dirname(a.summary_json) or ".", exist_ok=True)
    with open(a.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{a.name}] objective={a.objective}")
    print(f"best thresholds: t1={best['t1']:.3f}, t3={best['t3']:.3f}, t5={best['t5']:.3f}")
    print(f"val  base qwk/mae: {base_val['qwk']:.3f}/{base_val['mae']:.3f} -> tuned: {tuned_val['qwk']:.3f}/{tuned_val['mae']:.3f}")
    print(f"test base qwk/mae: {base_test['qwk']:.3f}/{base_test['mae']:.3f} -> tuned: {tuned_test['qwk']:.3f}/{tuned_test['mae']:.3f}")
    print(f"summary: {a.summary_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="method")
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--objective", type=str, default="qwk", choices=["qwk", "mae"])
    ap.add_argument("--start", type=float, default=0.1)
    ap.add_argument("--end", type=float, default=0.9)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--out_val_csv", required=True)
    ap.add_argument("--out_test_csv", required=True)
    ap.add_argument("--summary_json", required=True)
    args = ap.parse_args()
    main(args)
