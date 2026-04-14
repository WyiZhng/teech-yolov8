import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


SPLIT_NAMES = ["train", "val", "test"]
TRAIN_IDX, VAL_IDX, TEST_IDX = 0, 1, 2
B_CLASS, C_CLASS = 2, 3


@dataclass
class ScoreConfig:
    c_min: int = 8
    b_min: int = 30
    target_b_train: int = -1
    target_b_val: int = -1
    target_b_test: int = -1
    target_c_train: int = -1
    target_c_val: int = -1
    target_c_test: int = -1
    target_weight: float = 40.0


def map_ic4(icdas: int) -> int:
    if icdas <= 0:
        return 0
    if icdas <= 2:
        return 1
    if icdas <= 4:
        return 2
    return 3


def resolve_group_series(df: pd.DataFrame):
    if "patient_id" in df.columns and df["patient_id"].notna().any():
        pid = df["patient_id"].astype("string")
        missing = pid.isna() | (pid.str.strip() == "")
        group = pid.fillna("")
        group = group.where(~missing, "__img__" + df["image_id"].astype(str))
        return group.astype(str), "patient_id"
    return df["image_id"].astype(str), "image_id"


def class_series(df: pd.DataFrame) -> pd.Series:
    if "ic4" in df.columns:
        return df["ic4"].astype(int)
    if "icdas" in df.columns:
        return df["icdas"].astype(int).map(map_ic4)
    raise ValueError("Input CSVs must contain ic4 or icdas column")


def summarize_split_counts(split_class_counts: np.ndarray, split_roi_counts: np.ndarray, split_group_counts: np.ndarray):
    out = {}
    for s_idx, s_name in enumerate(SPLIT_NAMES):
        out[s_name] = {
            "groups": int(split_group_counts[s_idx]),
            "rois": int(split_roi_counts[s_idx]),
            "class_0": int(split_class_counts[s_idx, 0]),
            "class_1": int(split_class_counts[s_idx, 1]),
            "class_2_B": int(split_class_counts[s_idx, B_CLASS]),
            "class_3_C": int(split_class_counts[s_idx, C_CLASS]),
        }
    return out


def hard_deficit(split_class_counts: np.ndarray, cfg: ScoreConfig) -> int:
    b_val = int(split_class_counts[VAL_IDX, B_CLASS])
    b_test = int(split_class_counts[TEST_IDX, B_CLASS])
    c_val = int(split_class_counts[VAL_IDX, C_CLASS])
    c_test = int(split_class_counts[TEST_IDX, C_CLASS])
    return (
        max(0, cfg.c_min - c_val)
        + max(0, cfg.c_min - c_test)
        + max(0, cfg.b_min - b_val)
        + max(0, cfg.b_min - b_test)
    )


def target_deviation_penalty(split_class_counts: np.ndarray, cfg: ScoreConfig) -> float:
    targets = [
        (TRAIN_IDX, B_CLASS, cfg.target_b_train),
        (VAL_IDX, B_CLASS, cfg.target_b_val),
        (TEST_IDX, B_CLASS, cfg.target_b_test),
        (TRAIN_IDX, C_CLASS, cfg.target_c_train),
        (VAL_IDX, C_CLASS, cfg.target_c_val),
        (TEST_IDX, C_CLASS, cfg.target_c_test),
    ]
    sq_sum = 0.0
    for s_idx, c_idx, tgt in targets:
        if tgt >= 0:
            cur = float(split_class_counts[s_idx, c_idx])
            sq_sum += (cur - float(tgt)) ** 2
    return cfg.target_weight * sq_sum


def penalty(split_class_counts: np.ndarray, split_roi_counts: np.ndarray, mode: str, cfg: ScoreConfig) -> float:
    b_val = int(split_class_counts[VAL_IDX, B_CLASS])
    b_test = int(split_class_counts[TEST_IDX, B_CLASS])
    c_val = int(split_class_counts[VAL_IDX, C_CLASS])
    c_test = int(split_class_counts[TEST_IDX, C_CLASS])

    min_c = min(c_val, c_test)
    min_b = min(b_val, b_test)
    target_pen = target_deviation_penalty(split_class_counts, cfg)

    if mode == "constraint":
        deficit = hard_deficit(split_class_counts, cfg)
        return (
            deficit * 1_000_000.0
            - 5000.0 * min_c
            - 100.0 * min_b
            + 30.0 * abs(c_val - c_test)
            + 2.0 * abs(b_val - b_test)
            + 0.001 * abs(int(split_roi_counts[VAL_IDX]) - int(split_roi_counts[TEST_IDX]))
            + target_pen
        )

    if mode == "fallback":
        return (
            -10000.0 * min_c
            -200.0 * min_b
            + 20.0 * abs(c_val - c_test)
            + 2.0 * abs(b_val - b_test)
            + 0.001 * abs(int(split_roi_counts[VAL_IDX]) - int(split_roi_counts[TEST_IDX]))
            + target_pen
        )

    raise ValueError(f"Unknown mode: {mode}")


def build_initial_assignment(n_groups: int, target_counts, rng: np.random.Generator) -> np.ndarray:
    labels = np.empty(n_groups, dtype=np.int8)
    perm = rng.permutation(n_groups)
    a, b, c = target_counts
    labels[perm[:a]] = TRAIN_IDX
    labels[perm[a:a + b]] = VAL_IDX
    labels[perm[a + b:a + b + c]] = TEST_IDX
    return labels


def compute_split_stats(labels: np.ndarray, group_class_counts: np.ndarray, group_roi_counts: np.ndarray):
    split_class_counts = np.zeros((3, 4), dtype=np.int64)
    split_roi_counts = np.zeros(3, dtype=np.int64)
    split_group_counts = np.zeros(3, dtype=np.int64)
    for i, s in enumerate(labels):
        split_class_counts[s] += group_class_counts[i]
        split_roi_counts[s] += group_roi_counts[i]
        split_group_counts[s] += 1
    return split_class_counts, split_roi_counts, split_group_counts


def optimize_one_start(labels_init: np.ndarray,
                       group_class_counts: np.ndarray,
                       group_roi_counts: np.ndarray,
                       swap_attempts: int,
                       mode: str,
                       cfg: ScoreConfig,
                       rng: np.random.Generator):
    labels = labels_init.copy()
    split_class_counts, split_roi_counts, split_group_counts = compute_split_stats(labels, group_class_counts, group_roi_counts)
    cur_pen = penalty(split_class_counts, split_roi_counts, mode=mode, cfg=cfg)

    best_labels = labels.copy()
    best_pen = cur_pen
    best_split_class = split_class_counts.copy()
    best_split_roi = split_roi_counts.copy()

    for _ in range(swap_attempts):
        s1, s2 = rng.choice(3, size=2, replace=False)
        idx1_cands = np.flatnonzero(labels == s1)
        idx2_cands = np.flatnonzero(labels == s2)
        if len(idx1_cands) == 0 or len(idx2_cands) == 0:
            continue

        i1 = int(rng.choice(idx1_cands))
        i2 = int(rng.choice(idx2_cands))

        g1_cls = group_class_counts[i1]
        g2_cls = group_class_counts[i2]
        g1_roi = int(group_roi_counts[i1])
        g2_roi = int(group_roi_counts[i2])

        split_class_counts[s1] -= g1_cls
        split_class_counts[s1] += g2_cls
        split_class_counts[s2] -= g2_cls
        split_class_counts[s2] += g1_cls

        split_roi_counts[s1] = split_roi_counts[s1] - g1_roi + g2_roi
        split_roi_counts[s2] = split_roi_counts[s2] - g2_roi + g1_roi

        new_pen = penalty(split_class_counts, split_roi_counts, mode=mode, cfg=cfg)

        if new_pen <= cur_pen:
            labels[i1], labels[i2] = labels[i2], labels[i1]
            cur_pen = new_pen

            if new_pen < best_pen:
                best_pen = new_pen
                best_labels = labels.copy()
                best_split_class = split_class_counts.copy()
                best_split_roi = split_roi_counts.copy()
        else:
            split_class_counts[s1] += g1_cls
            split_class_counts[s1] -= g2_cls
            split_class_counts[s2] += g2_cls
            split_class_counts[s2] -= g1_cls
            split_roi_counts[s1] = split_roi_counts[s1] + g1_roi - g2_roi
            split_roi_counts[s2] = split_roi_counts[s2] + g2_roi - g1_roi

    best_split_group = np.array([
        int((best_labels == TRAIN_IDX).sum()),
        int((best_labels == VAL_IDX).sum()),
        int((best_labels == TEST_IDX).sum()),
    ], dtype=np.int64)

    return best_labels, best_pen, best_split_class, best_split_roi, best_split_group


def run_search(group_class_counts: np.ndarray,
               group_roi_counts: np.ndarray,
               target_counts,
               n_initial: int,
               swap_attempts: int,
               mode: str,
               cfg: ScoreConfig,
               seed: int):
    rng = np.random.default_rng(seed)

    global_best = None
    global_best_pen = float("inf")
    global_best_class = None
    global_best_roi = None
    global_best_group = None

    for _ in range(n_initial):
        labels0 = build_initial_assignment(len(group_class_counts), target_counts, rng)
        labels, pen, split_class, split_roi, split_group = optimize_one_start(
            labels0,
            group_class_counts,
            group_roi_counts,
            swap_attempts=swap_attempts,
            mode=mode,
            cfg=cfg,
            rng=rng,
        )

        if pen < global_best_pen:
            global_best_pen = pen
            global_best = labels
            global_best_class = split_class
            global_best_roi = split_roi
            global_best_group = split_group

    return global_best, global_best_pen, global_best_class, global_best_roi, global_best_group


def save_outputs(df_all: pd.DataFrame,
                 group_series: pd.Series,
                 group_keys,
                 labels: np.ndarray,
                 out_dir: str,
                 prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    group_to_split = {g: SPLIT_NAMES[int(s)] for g, s in zip(group_keys, labels)}

    split_col = group_series.map(group_to_split)
    df_out = df_all.copy()
    df_out["new_split"] = split_col.values

    train_df = df_out[df_out["new_split"] == "train"].drop(columns=["new_split"])
    val_df = df_out[df_out["new_split"] == "val"].drop(columns=["new_split"])
    test_df = df_out[df_out["new_split"] == "test"].drop(columns=["new_split"])

    train_path = os.path.join(out_dir, f"{prefix}_train.csv")
    val_path = os.path.join(out_dir, f"{prefix}_val.csv")
    test_path = os.path.join(out_dir, f"{prefix}_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    mapping_df = pd.DataFrame({
        "group_id": list(group_to_split.keys()),
        "split": list(group_to_split.values()),
    })
    map_path = os.path.join(out_dir, f"{prefix}_group_split.csv")
    mapping_df.to_csv(map_path, index=False)

    return train_path, val_path, test_path, map_path


def main():
    ap = argparse.ArgumentParser(description="Rebuild ICDAS4 train/val/test split with strict group-wise constraints.")
    ap.add_argument("--in_train", type=str, default="icdas4_train.csv")
    ap.add_argument("--in_val", type=str, default="icdas4_val.csv")
    ap.add_argument("--in_test", type=str, default="icdas4_test.csv")

    ap.add_argument("--target_train_groups", type=int, default=207)
    ap.add_argument("--target_val_groups", type=int, default=34)
    ap.add_argument("--target_test_groups", type=int, default=35)

    ap.add_argument("--n_initial", type=int, default=120)
    ap.add_argument("--swap_attempts", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--c_min", type=int, default=8)
    ap.add_argument("--b_min", type=int, default=30)
    ap.add_argument("--target_b_train", type=int, default=-1)
    ap.add_argument("--target_b_val", type=int, default=-1)
    ap.add_argument("--target_b_test", type=int, default=-1)
    ap.add_argument("--target_c_train", type=int, default=-1)
    ap.add_argument("--target_c_val", type=int, default=-1)
    ap.add_argument("--target_c_test", type=int, default=-1)
    ap.add_argument("--target_weight", type=float, default=40.0)

    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--out_prefix", type=str, default="icdas4_rebalanced_75_12p5_12p5")

    args = ap.parse_args()

    df_train = pd.read_csv(args.in_train)
    df_val = pd.read_csv(args.in_val)
    df_test = pd.read_csv(args.in_test)
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    if "image_id" not in df_all.columns:
        raise ValueError("Input CSVs must contain image_id column")

    group_series, group_key_source = resolve_group_series(df_all)
    y = class_series(df_all)

    group_keys = pd.Index(group_series.unique())
    n_groups = len(group_keys)
    target_counts = (
        args.target_train_groups,
        args.target_val_groups,
        args.target_test_groups,
    )

    if sum(target_counts) != n_groups:
        raise ValueError(
            f"Target group counts {target_counts} sum to {sum(target_counts)}, but found {n_groups} groups."
        )

    group_index = pd.Series(np.arange(n_groups), index=group_keys)
    gid = group_series.map(group_index).to_numpy()

    group_class_counts = np.zeros((n_groups, 4), dtype=np.int64)
    group_roi_counts = np.zeros(n_groups, dtype=np.int64)

    for i, cls in zip(gid, y.to_numpy(dtype=np.int64)):
        group_class_counts[i, cls] += 1
        group_roi_counts[i] += 1

    cfg = ScoreConfig(
        c_min=args.c_min,
        b_min=args.b_min,
        target_b_train=args.target_b_train,
        target_b_val=args.target_b_val,
        target_b_test=args.target_b_test,
        target_c_train=args.target_c_train,
        target_c_val=args.target_c_val,
        target_c_test=args.target_c_test,
        target_weight=args.target_weight,
    )

    best_labels, best_pen, best_cls, best_roi, best_grp = run_search(
        group_class_counts,
        group_roi_counts,
        target_counts=target_counts,
        n_initial=max(args.n_initial, 100),
        swap_attempts=max(args.swap_attempts, 3000),
        mode="constraint",
        cfg=cfg,
        seed=args.seed,
    )

    deficit = hard_deficit(best_cls, cfg)
    downgraded = False
    mode_used = "constraint"

    if deficit > 0:
        downgraded = True
        mode_used = "fallback"
        best_labels, best_pen, best_cls, best_roi, best_grp = run_search(
            group_class_counts,
            group_roi_counts,
            target_counts=target_counts,
            n_initial=max(args.n_initial, 100),
            swap_attempts=max(args.swap_attempts, 3000),
            mode="fallback",
            cfg=cfg,
            seed=args.seed + 97,
        )

    train_path, val_path, test_path, map_path = save_outputs(
        df_all=df_all,
        group_series=group_series,
        group_keys=group_keys,
        labels=best_labels,
        out_dir=args.out_dir,
        prefix=args.out_prefix,
    )

    split_summary = summarize_split_counts(best_cls, best_roi, best_grp)

    train_groups = set(group_keys[np.where(best_labels == TRAIN_IDX)[0]])
    val_groups = set(group_keys[np.where(best_labels == VAL_IDX)[0]])
    test_groups = set(group_keys[np.where(best_labels == TEST_IDX)[0]])

    leakage_ok = (
        len(train_groups & val_groups) == 0
        and len(train_groups & test_groups) == 0
        and len(val_groups & test_groups) == 0
    )

    report = {
        "group_key_source": group_key_source,
        "total_groups": int(n_groups),
        "target_group_counts": {
            "train": int(target_counts[0]),
            "val": int(target_counts[1]),
            "test": int(target_counts[2]),
        },
        "search": {
            "n_initial": int(max(args.n_initial, 100)),
            "swap_attempts_per_initial": int(max(args.swap_attempts, 3000)),
            "seed": int(args.seed),
            "mode_used": mode_used,
            "downgraded": bool(downgraded),
            "constraint": {"C_min_each_of_val_test": int(args.c_min), "B_min_each_of_val_test": int(args.b_min)},
            "targets": {
                "B_train": int(args.target_b_train),
                "B_val": int(args.target_b_val),
                "B_test": int(args.target_b_test),
                "C_train": int(args.target_c_train),
                "C_val": int(args.target_c_val),
                "C_test": int(args.target_c_test),
                "target_weight": float(args.target_weight),
            },
        },
        "penalty": float(best_pen),
        "hard_deficit_after": int(hard_deficit(best_cls, cfg)),
        "split_summary": split_summary,
        "leakage_check": {
            "train_val_overlap": int(len(train_groups & val_groups)),
            "train_test_overlap": int(len(train_groups & test_groups)),
            "val_test_overlap": int(len(val_groups & test_groups)),
            "passed": bool(leakage_ok),
        },
        "outputs": {
            "train_csv": train_path,
            "val_csv": val_path,
            "test_csv": test_path,
            "group_mapping_csv": map_path,
        },
    }

    report_path = os.path.join(args.out_dir, f"{args.out_prefix}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
