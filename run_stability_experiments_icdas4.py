import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score


REQUIRED_METHODS = [
    "ord2seq_guided_softmax_ordplus",
    "softmax_baseline",
    "corn",
    "existing_ordinal_ord2seq",
]

OPTIONAL_METHODS = [
    "coral",
    "coral_strict",
    "dch_ordinal",
    "your_own_ordinal_masked",
]

METHOD_SPECS = {
    "ord2seq_guided_softmax_ordplus": {
        "label": "Ord2Seq-guided Softmax OrdPlus",
        "train_script": "train_softmax_ordplus_icdas4.py",
        "eval_script": "eval_softmax_ordplus_on_roi_icdas4.py",
        "ckpt_template": "softmax_ordplus_icdas4_seed{seed}.pt",
    },
    "softmax_baseline": {
        "label": "Softmax baseline",
        "train_script": "train_softmax_head_icdas4.py",
        "eval_script": "tools/eval_ordinal_on_roi_icdas4.py",
        "ckpt_template": "softmax_head_icdas4_seed{seed}.pt",
    },
    "corn": {
        "label": "CORN",
        "train_script": "train_corn_head_icdas4.py",
        "eval_script": "eval_corn_on_roi_icdas4.py",
        "ckpt_template": "corn_head_icdas4_seed{seed}.pt",
    },
    "existing_ordinal_ord2seq": {
        "label": "Existing Ordinal (Ord2Seq)",
        "train_script": "train_ordinal_head_min.py",
        "eval_script": "tools/eval_ordinal_on_roi_icdas4.py",
        "ckpt_template": "ord2seq_head_icdas4_seed{seed}.pt",
    },
    "coral": {
        "label": "CORAL (independent 3-logit)",
        "train_script": "train_coral_head_icdas4.py",
        "eval_script": "eval_coral_on_roi_icdas4.py",
        "ckpt_template": "coral_head_icdas4_seed{seed}.pt",
    },
    "coral_strict": {
        "label": "CORAL strict (shared-weight + bias)",
        "train_script": "train_coral_strict_head_icdas4.py",
        "eval_script": "eval_coral_strict_on_roi_icdas4.py",
        "ckpt_template": "coral_strict_head_icdas4_seed{seed}.pt",
    },
    "dch_ordinal": {
        "label": "DCH-Ordinal / DCH dynamic",
        "train_script": "train_dch_ordinal_head_icdas4.py",
        "eval_script": "eval_dch_ordinal_on_roi_icdas4.py",
        "ckpt_template": "dch_ordinal_head_icdas4_seed{seed}.pt",
    },
    "your_own_ordinal_masked": {
        "label": "Your own Ordinal (masked)",
        "train_script": "train_ordinal_head_min.py",
        "eval_script": "tools/eval_ordinal_on_roi_icdas4.py",
        "ckpt_template": "ordinal_head_icdas4_seed{seed}.pt",
    },
}


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_log(log_path, text):
    line = f"[{now_str()}] {text}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def safe_auc(y_true_bin, y_score):
    y_true_bin = np.asarray(y_true_bin)
    y_score = np.asarray(y_score)
    if len(y_true_bin) == 0:
        return np.nan
    if (y_true_bin == 1).sum() > 0 and (y_true_bin == 0).sum() > 0:
        return float(roc_auc_score(y_true_bin, y_score))
    return np.nan


def map_ic4(icdas):
    if icdas <= 0:
        return 0
    if icdas <= 2:
        return 1
    if icdas <= 4:
        return 2
    return 3


def _get_gt_ic4(df):
    if "gt_class" in df.columns:
        return df["gt_class"].to_numpy(dtype=np.int64)
    if "ic4" in df.columns:
        return df["ic4"].to_numpy(dtype=np.int64)
    if "y_gt" in df.columns:
        return df["y_gt"].to_numpy(dtype=np.int64)
    if "icdas" in df.columns:
        return df["icdas"].apply(map_ic4).to_numpy(dtype=np.int64)
    raise ValueError("Prediction CSV missing GT class columns: gt_class/ic4/y_gt/icdas")


def _get_pred_ic4(df):
    if "pred_class" in df.columns:
        return df["pred_class"].to_numpy(dtype=np.int64)
    if "pred_ic4" in df.columns:
        return df["pred_ic4"].to_numpy(dtype=np.int64)
    if "y_pred" in df.columns:
        return df["y_pred"].to_numpy(dtype=np.int64)
    if {"p_ge1", "p_ge3", "p_ge5"}.issubset(df.columns):
        p1 = df["p_ge1"].to_numpy()
        p3 = df["p_ge3"].to_numpy()
        p5 = df["p_ge5"].to_numpy()
        pred = np.zeros(len(df), dtype=np.int64)
        pred = np.where(p1 >= 0.5, 1, pred)
        pred = np.where(p3 >= 0.5, 2, pred)
        pred = np.where(p5 >= 0.5, 3, pred)
        return pred
    raise ValueError("Prediction CSV missing prediction columns: pred_class/pred_ic4/y_pred or p_ge1/3/5")


def _get_p_ge(df, ge_name):
    if ge_name in df.columns:
        return df[ge_name].to_numpy(dtype=np.float64)
    # Fallback for softmax-style columns.
    if ge_name == "p_ge1" and {"pA", "pB", "pC"}.issubset(df.columns):
        return (df["pA"] + df["pB"] + df["pC"]).to_numpy(dtype=np.float64)
    if ge_name == "p_ge3" and {"pB", "pC"}.issubset(df.columns):
        return (df["pB"] + df["pC"]).to_numpy(dtype=np.float64)
    if ge_name == "p_ge5" and "pC" in df.columns:
        return df["pC"].to_numpy(dtype=np.float64)
    raise ValueError(f"Prediction CSV missing {ge_name} (or softmax fallback columns)")


def compute_metrics_from_pred_csv(pred_csv_path):
    df = pd.read_csv(pred_csv_path)
    gt = _get_gt_ic4(df)
    pred = _get_pred_ic4(df)
    p_ge1 = _get_p_ge(df, "p_ge1")
    p_ge3 = _get_p_ge(df, "p_ge3")
    p_ge5 = _get_p_ge(df, "p_ge5")

    mae = float(np.mean(np.abs(gt - pred)))
    qwk = float(cohen_kappa_score(gt, pred, weights="quadratic"))

    auc_ge1 = safe_auc((gt >= 1).astype(int), p_ge1)
    auc_ge3 = safe_auc((gt >= 2).astype(int), p_ge3)
    auc_ge5 = safe_auc((gt >= 3).astype(int), p_ge5)

    return {
        "mae": mae,
        "qwk": qwk,
        "auc_ge1": auc_ge1,
        "auc_ge3": auc_ge3,
        "auc_ge5": auc_ge5,
    }


def run_cmd(cmd, log_path):
    write_log(log_path, "RUN: " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(p.stdout)
        if not p.stdout.endswith("\n"):
            f.write("\n")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def save_raw_rows(raw_rows, raw_csv_path):
    pd.DataFrame(raw_rows).to_csv(raw_csv_path, index=False)


def build_train_cmd(args, method_key, seed, ckpt_path):
    py = list(args.python_exec_cmd)
    common = [
        "--train_csv", args.train_csv,
        "--val_csv", args.val_csv,
        "--img_root_train", args.img_root_train,
        "--img_root_val", args.img_root_val,
        "--img_size", str(args.img_size),
        "--expand", str(args.expand),
        "--bs", str(args.bs),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--out", ckpt_path,
    ]

    spec = METHOD_SPECS[method_key]
    script = spec["train_script"]

    if method_key == "ord2seq_guided_softmax_ordplus":
        return [
            *py,
            script,
            "--train_csv", args.train_csv,
            "--val_csv", args.val_csv,
            "--img_root_train", args.img_root_train,
            "--img_root_val", args.img_root_val,
            "--img_size", str(args.img_size),
            "--expand", str(args.expand),
            "--bs", str(args.bs),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--workers", str(args.workers),
            "--seed", str(seed),
            "--ord_mode", "ord2seq",
            "--out", ckpt_path,
        ]

    if method_key == "softmax_baseline":
        return [*py, script, *common, "--head_type", "softmax", "--seed", str(seed), "--workers", str(args.workers)] + (
            ["--deterministic"] if args.deterministic else []
        )

    if method_key == "corn":
        return [*py, script, *common, "--seed", str(seed), "--workers", str(args.workers)] + (
            ["--deterministic"] if args.deterministic else []
        )

    if method_key == "existing_ordinal_ord2seq":
        return [
            *py,
            script,
            *common,
            "--head_type", "ord2seq",
            "--seed", str(seed),
            "--workers", str(args.workers),
        ] + (["--deterministic"] if args.deterministic else [])

    if method_key == "coral":
        return [*py, script, *common]

    if method_key == "coral_strict":
        return [*py, script, *common]

    if method_key == "dch_ordinal":
        return [*py, script, *common, "--seed", str(seed)]

    if method_key == "your_own_ordinal_masked":
        return [
            *py,
            script,
            *common,
            "--head_type", "masked",
            "--seed", str(seed),
            "--workers", str(args.workers),
        ] + (["--deterministic"] if args.deterministic else [])

    raise ValueError(f"Unknown method: {method_key}")


def build_eval_commands(args, method_key, ckpt_path, val_pred_csv, test_pred_csv):
    py = list(args.python_exec_cmd)
    spec = METHOD_SPECS[method_key]
    script = spec["eval_script"]

    if method_key in ["softmax_baseline", "existing_ordinal_ord2seq", "your_own_ordinal_masked"]:
        val_cmd = [
            *py,
            script,
            "--val_csv", args.val_csv,
            "--img_root", args.img_root_val,
            "--ckpt", ckpt_path,
            "--img_size", str(args.img_size),
            "--expand", str(args.expand),
            "--bs", str(args.eval_bs),
            "--out_csv", val_pred_csv,
        ]
        test_cmd = [
            *py,
            script,
            "--val_csv", args.test_csv,
            "--img_root", args.img_root_test,
            "--ckpt", ckpt_path,
            "--img_size", str(args.img_size),
            "--expand", str(args.expand),
            "--bs", str(args.eval_bs),
            "--out_csv", test_pred_csv,
        ]
        return [val_cmd, test_cmd]

    if method_key == "ord2seq_guided_softmax_ordplus":
        val_cmd = [
            *py,
            script,
            "--val_csv", args.val_csv,
            "--img_root", args.img_root_val,
            "--ckpt", ckpt_path,
            "--img_size", str(args.img_size),
            "--expand", str(args.expand),
            "--bs", str(args.eval_bs),
            "--out_csv", val_pred_csv,
        ]
        test_cmd = [
            *py,
            script,
            "--val_csv", args.test_csv,
            "--img_root", args.img_root_test,
            "--ckpt", ckpt_path,
            "--img_size", str(args.img_size),
            "--expand", str(args.expand),
            "--bs", str(args.eval_bs),
            "--out_csv", test_pred_csv,
        ]
        return [val_cmd, test_cmd]

    if method_key in ["corn", "coral", "coral_strict", "dch_ordinal"]:
        one_cmd = [
            *py,
            script,
            "--val_csv", args.val_csv,
            "--test_csv", args.test_csv,
            "--img_root_val", args.img_root_val,
            "--img_root_test", args.img_root_test,
            "--ckpt", ckpt_path,
            "--img_size", str(args.img_size),
            "--expand", str(args.expand),
            "--bs", str(args.eval_bs),
            "--out_val_csv", val_pred_csv,
            "--out_test_csv", test_pred_csv,
        ]
        return [one_cmd]

    raise ValueError(f"Unknown method for eval: {method_key}")


def summarize_runs(raw_df):
    grouped = (
        raw_df.groupby(["method", "split"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            qwk_mean=("qwk", "mean"),
            qwk_std=("qwk", "std"),
            auc_ge1_mean=("auc_ge1", "mean"),
            auc_ge1_std=("auc_ge1", "std"),
            auc_ge3_mean=("auc_ge3", "mean"),
            auc_ge3_std=("auc_ge3", "std"),
            auc_ge5_mean=("auc_ge5", "mean"),
            auc_ge5_std=("auc_ge5", "std"),
        )
    )
    return grouped


def fmt_ms(mean_v, std_v):
    return f"{mean_v:.3f}±{std_v:.3f}"


def dataframe_to_markdown_table(df):
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_rank_table(df_split):
    t = df_split.sort_values(["qwk_mean", "mae_mean"], ascending=[False, True]).reset_index(drop=True).copy()
    t.insert(0, "Rank", np.arange(1, len(t) + 1))
    t["MAE (mean±std)"] = [fmt_ms(m, s) for m, s in zip(t["mae_mean"], t["mae_std"])]
    t["QWK (mean±std)"] = [fmt_ms(m, s) for m, s in zip(t["qwk_mean"], t["qwk_std"])]
    t["AUC>=1"] = [fmt_ms(m, s) for m, s in zip(t["auc_ge1_mean"], t["auc_ge1_std"])]
    t["AUC>=3"] = [fmt_ms(m, s) for m, s in zip(t["auc_ge3_mean"], t["auc_ge3_std"])]
    t["AUC>=5"] = [fmt_ms(m, s) for m, s in zip(t["auc_ge5_mean"], t["auc_ge5_std"])]
    return t[["Rank", "method", "MAE (mean±std)", "QWK (mean±std)", "AUC>=1", "AUC>=3", "AUC>=5"]]


def write_paper_markdown(summary_df, out_md, seeds):
    val_df = summary_df[summary_df["split"] == "val"].copy()
    test_df = summary_df[summary_df["split"] == "test"].copy()

    test_rank = make_rank_table(test_df)
    val_rank = make_rank_table(val_df)

    soft = test_df[test_df["method"] == "Softmax baseline"]
    main = test_df[test_df["method"] == "Ord2Seq-guided Softmax OrdPlus"]
    compare_rows = []
    if len(soft) == 1 and len(main) == 1:
        soft = soft.iloc[0]
        main = main.iloc[0]
        compare_rows.append(
            {
                "Method": "Ord2Seq-guided Softmax OrdPlus",
                "Test MAE mean±std": fmt_ms(main["mae_mean"], main["mae_std"]),
                "Test QWK mean±std": fmt_ms(main["qwk_mean"], main["qwk_std"]),
                "ΔQWK vs Softmax": f"{(main['qwk_mean'] - soft['qwk_mean']):+.3f}",
                "ΔMAE vs Softmax": f"{(main['mae_mean'] - soft['mae_mean']):+.3f}",
            }
        )
        compare_rows.append(
            {
                "Method": "Softmax baseline",
                "Test MAE mean±std": fmt_ms(soft["mae_mean"], soft["mae_std"]),
                "Test QWK mean±std": fmt_ms(soft["qwk_mean"], soft["qwk_std"]),
                "ΔQWK vs Softmax": "+0.000",
                "ΔMAE vs Softmax": "+0.000",
            }
        )

    lines = []
    lines.append("# ICDAS4 ROI Stability Summary (3 Seeds)")
    lines.append("")
    lines.append("## Protocol")
    lines.append("- Same split/preprocess/backbone/training-budget across methods.")
    lines.append("- Fixed threshold decoding at default settings (no per-method post-hoc tuning).")
    lines.append("- No proposal-stage changes.")
    lines.append("- ResNet18 backbone and existing method-specific scripts are reused.")
    lines.append("- Only random seed is varied.")
    lines.append(f"- Seeds: {', '.join(str(x) for x in seeds)}")
    lines.append("")

    lines.append("## Table 1: Test stability leaderboard")
    lines.append(dataframe_to_markdown_table(test_rank.rename(columns={"method": "Method"})))
    lines.append("")

    lines.append("## Table 2: Validation stability leaderboard")
    lines.append(dataframe_to_markdown_table(val_rank.rename(columns={"method": "Method"})))
    lines.append("")

    lines.append("## Table 3: Main method vs Softmax baseline")
    if compare_rows:
        lines.append(dataframe_to_markdown_table(pd.DataFrame(compare_rows)))
    else:
        lines.append("Insufficient rows for comparison (main or softmax missing).")
    lines.append("")

    lines.append("## Brief Stability Notes")
    qwk_std_test = test_df[["method", "qwk_std"]].sort_values("qwk_std", ascending=False)
    if len(qwk_std_test) > 0:
        top_var = qwk_std_test.iloc[0]
        lines.append(
            f"- Largest test QWK fluctuation is {top_var['method']} with std={top_var['qwk_std']:.3f}."
        )
    mae_std_test = test_df[["method", "mae_std"]].sort_values("mae_std", ascending=False)
    if len(mae_std_test) > 0:
        top_mae = mae_std_test.iloc[0]
        lines.append(
            f"- Largest test MAE fluctuation is {top_mae['method']} with std={top_mae['mae_std']:.3f}."
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Run ICDAS4 ROI stability experiments with 3 fixed seeds.")
    ap.add_argument("--methods", nargs="+", default=REQUIRED_METHODS)
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])

    ap.add_argument("--train_csv", type=str, default="icdas4_train.csv")
    ap.add_argument("--val_csv", type=str, default="icdas4_val.csv")
    ap.add_argument("--test_csv", type=str, default="icdas4_test.csv")

    ap.add_argument("--img_root_train", type=str, required=True)
    ap.add_argument("--img_root_val", type=str, required=True)
    ap.add_argument("--img_root_test", type=str, required=True)

    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--expand", type=float, default=1.25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--python_exec", type=str, default=sys.executable)
    ap.add_argument("--skip_if_ckpt_exists", action="store_true")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--ckpt_dir", type=str, default="ckpt/stability")
    ap.add_argument("--pred_dir", type=str, default="pred_csv/stability")

    ap.add_argument("--raw_csv", type=str, default="stability_runs_raw.csv")
    ap.add_argument("--summary_csv", type=str, default="stability_summary_mean_std.csv")
    ap.add_argument("--paper_md", type=str, default="stability_summary_for_paper.md")
    ap.add_argument("--log_file", type=str, default="stability_run_log.txt")

    args = ap.parse_args()
    args.python_exec_cmd = shlex.split(args.python_exec)

    methods = list(args.methods)
    if args.include_optional:
        methods.extend(OPTIONAL_METHODS)
    methods = [m for m in methods if m in METHOD_SPECS]

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    if args.resume and os.path.exists(args.raw_csv):
        old_raw = pd.read_csv(args.raw_csv)
        raw_rows = old_raw.to_dict(orient="records")
        existing_ok = {
            (r["method"], int(r["seed"]), r["split"])
            for r in raw_rows
            if r.get("status") == "ok"
        }
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{now_str()}] Resume stability run\n")
    else:
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(f"[{now_str()}] Start stability run\n")
        raw_rows = []
        existing_ok = set()

    write_log(args.log_file, f"Methods: {methods}")
    write_log(args.log_file, f"Seeds: {args.seeds}")

    for method_key in methods:
        spec = METHOD_SPECS[method_key]
        method_name = spec["label"]

        for seed in args.seeds:
            ckpt_name = spec["ckpt_template"].format(seed=seed)
            ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

            val_pred_csv = os.path.join(args.pred_dir, f"roi_val_icdas4_{method_key}_seed{seed}.csv")
            test_pred_csv = os.path.join(args.pred_dir, f"roi_test_icdas4_{method_key}_seed{seed}.csv")

            train_cmd = build_train_cmd(args, method_key, seed, ckpt_path)
            eval_cmds = build_eval_commands(args, method_key, ckpt_path, val_pred_csv, test_pred_csv)

            if (method_name, seed, "val") in existing_ok and (method_name, seed, "test") in existing_ok:
                write_log(args.log_file, f"Skip completed run: method={method_name} seed={seed}")
                continue

            try:
                if args.skip_if_ckpt_exists and os.path.exists(ckpt_path):
                    write_log(args.log_file, f"Skip train (checkpoint exists): {ckpt_path}")
                else:
                    run_cmd(train_cmd, args.log_file)

                for cmd in eval_cmds:
                    run_cmd(cmd, args.log_file)

                val_metrics = compute_metrics_from_pred_csv(val_pred_csv)
                test_metrics = compute_metrics_from_pred_csv(test_pred_csv)

                for split, metrics, out_csv in [
                    ("val", val_metrics, val_pred_csv),
                    ("test", test_metrics, test_pred_csv),
                ]:
                    row = {
                        "method": method_name,
                        "seed": seed,
                        "split": split,
                        "mae": metrics["mae"],
                        "qwk": metrics["qwk"],
                        "auc_ge1": metrics["auc_ge1"],
                        "auc_ge3": metrics["auc_ge3"],
                        "auc_ge5": metrics["auc_ge5"],
                        "checkpoint_path": ckpt_path,
                        "train_script": spec["train_script"],
                        "eval_script": spec["eval_script"],
                        "pred_csv": out_csv,
                        "status": "ok",
                        "error": "",
                    }
                    raw_rows.append(row)
                    save_raw_rows(raw_rows, args.raw_csv)
                    write_log(
                        args.log_file,
                        f"DONE method={method_name} seed={seed} split={split} "
                        f"MAE={row['mae']:.4f} QWK={row['qwk']:.4f} "
                        f"AUCs={row['auc_ge1']:.4f}/{row['auc_ge3']:.4f}/{row['auc_ge5']:.4f}",
                    )
            except Exception as e:
                err = str(e)
                write_log(args.log_file, f"FAILED method={method_name} seed={seed}: {err}")
                for split in ["val", "test"]:
                    raw_rows.append(
                        {
                            "method": method_name,
                            "seed": seed,
                            "split": split,
                            "mae": np.nan,
                            "qwk": np.nan,
                            "auc_ge1": np.nan,
                            "auc_ge3": np.nan,
                            "auc_ge5": np.nan,
                            "checkpoint_path": ckpt_path,
                            "train_script": spec["train_script"],
                            "eval_script": spec["eval_script"],
                            "pred_csv": "",
                            "status": "failed",
                            "error": err,
                        }
                    )
                    save_raw_rows(raw_rows, args.raw_csv)

    raw_df = pd.DataFrame(raw_rows)
    if len(raw_df) > 0:
        # Keep one final row per (method, seed, split), preferring successful runs.
        raw_df["_status_rank"] = raw_df["status"].map({"ok": 1, "failed": 0}).fillna(-1)
        raw_df = (
            raw_df.sort_values(["method", "seed", "split", "_status_rank"])
            .drop_duplicates(subset=["method", "seed", "split"], keep="last")
            .drop(columns=["_status_rank"])
            .reset_index(drop=True)
        )
    raw_df.to_csv(args.raw_csv, index=False)
    write_log(args.log_file, f"Saved raw runs: {args.raw_csv}")

    ok_df = raw_df[raw_df["status"] == "ok"].copy()
    if len(ok_df) == 0:
        write_log(args.log_file, "No successful runs. Skip summary generation.")
        return

    summary_df = summarize_runs(ok_df)
    summary_df.to_csv(args.summary_csv, index=False)
    write_log(args.log_file, f"Saved summary csv: {args.summary_csv}")

    write_paper_markdown(summary_df, args.paper_md, args.seeds)
    write_log(args.log_file, f"Saved paper markdown: {args.paper_md}")


if __name__ == "__main__":
    main()
