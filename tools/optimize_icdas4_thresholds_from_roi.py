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


def _is_better(cand, best, objective="qwk", lambda_mae=0.30):
    if objective == "mae":
        # primary lower MAE, tie-break higher QWK
        return (cand["mae"] < best["mae"]) or (
            abs(cand["mae"] - best["mae"]) < 1e-12 and cand["qwk"] > best["qwk"]
        )

    if objective == "qwk_mae":
        # maximize (QWK - lambda * MAE), tie-break higher QWK then lower MAE
        s_c = cand["qwk"] - lambda_mae * cand["mae"]
        s_b = best["qwk"] - lambda_mae * best["mae"]
        return (s_c > s_b) or (
            abs(s_c - s_b) < 1e-12
            and (
                (cand["qwk"] > best["qwk"])
                or (abs(cand["qwk"] - best["qwk"]) < 1e-12 and cand["mae"] < best["mae"])
            )
        )

    # default qwk: primary higher QWK, tie-break lower MAE
    return (cand["qwk"] > best["qwk"]) or (
        abs(cand["qwk"] - best["qwk"]) < 1e-12 and cand["mae"] < best["mae"]
    )


def _pareto_front(cands):
    # non-dominated set: maximize qwk, minimize mae
    front = []
    for i, a in enumerate(cands):
        dominated = False
        for j, b in enumerate(cands):
            if i == j:
                continue
            b_not_worse = (b["qwk"] >= a["qwk"]) and (b["mae"] <= a["mae"])
            b_strict_better = (b["qwk"] > a["qwk"]) or (b["mae"] < a["mae"])
            if b_not_worse and b_strict_better:
                dominated = True
                break
        if not dominated:
            front.append(a)
    front.sort(key=lambda x: (-x["qwk"], x["mae"]))
    return front


def grid_search(
    y,
    p1,
    p3,
    p5,
    start=0.1,
    end=0.9,
    step=0.02,
    objective="qwk",
    lambda_mae=0.30,
):
    grid = np.arange(start, end + 1e-9, step)

    best = None
    all_cands = []
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
                all_cands.append(cand)
                if best is None:
                    best = cand
                    continue
                better = _is_better(cand, best, objective=objective, lambda_mae=lambda_mae)
                if better:
                    best = cand
    return best, all_cands


def main(a):
    val_df = pd.read_csv(a.val_csv)
    test_df = pd.read_csv(a.test_csv)

    yv, p1v, p3v, p5v = extract_gt_and_probs(val_df)
    yt, p1t, p3t, p5t = extract_gt_and_probs(test_df)

    base_val = evaluate(yv, p1v, p3v, p5v, 0.5, 0.5, 0.5)
    base_test = evaluate(yt, p1t, p3t, p5t, 0.5, 0.5, 0.5)

    best, all_cands = grid_search(
        yv,
        p1v,
        p3v,
        p5v,
        start=a.start,
        end=a.end,
        step=a.step,
        objective=a.objective,
        lambda_mae=a.lambda_mae,
    )

    front = _pareto_front(all_cands)
    score = best["qwk"] - a.lambda_mae * best["mae"]

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
        "lambda_mae": a.lambda_mae,
        "selection_score_qwk_minus_lambda_mae": score,
        "best_thresholds": {"t1": best["t1"], "t3": best["t3"], "t5": best["t5"]},
        "pareto_front_size": len(front),
        "pareto_front_top": front[: a.pareto_topk],
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

    if a.pareto_json:
        os.makedirs(os.path.dirname(a.pareto_json) or ".", exist_ok=True)
        with open(a.pareto_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": a.name,
                    "objective": a.objective,
                    "lambda_mae": a.lambda_mae,
                    "pareto_front_size": len(front),
                    "pareto_front": front,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    os.makedirs(os.path.dirname(a.summary_json) or ".", exist_ok=True)
    with open(a.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{a.name}] objective={a.objective} lambda_mae={a.lambda_mae:.3f}")
    print(f"best thresholds: t1={best['t1']:.3f}, t3={best['t3']:.3f}, t5={best['t5']:.3f}")
    print(f"selection score (qwk-lambda*mae): {score:.5f}")
    print(f"pareto front size: {len(front)}")
    print(f"val  base qwk/mae: {base_val['qwk']:.3f}/{base_val['mae']:.3f} -> tuned: {tuned_val['qwk']:.3f}/{tuned_val['mae']:.3f}")
    print(f"test base qwk/mae: {base_test['qwk']:.3f}/{base_test['mae']:.3f} -> tuned: {tuned_test['qwk']:.3f}/{tuned_test['mae']:.3f}")
    print(f"summary: {a.summary_json}")
    if a.pareto_json:
        print(f"pareto: {a.pareto_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="method")
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--objective", type=str, default="qwk", choices=["qwk", "mae", "qwk_mae"])
    ap.add_argument("--lambda_mae", type=float, default=0.30)
    ap.add_argument("--start", type=float, default=0.1)
    ap.add_argument("--end", type=float, default=0.9)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--pareto_json", type=str, default="")
    ap.add_argument("--pareto_topk", type=int, default=20)
    ap.add_argument("--out_val_csv", required=True)
    ap.add_argument("--out_test_csv", required=True)
    ap.add_argument("--summary_json", required=True)
    args = ap.parse_args()
    main(args)
