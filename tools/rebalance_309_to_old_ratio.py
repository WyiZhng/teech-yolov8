#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image


SPLITS = ["train", "val", "test"]
GROUPS = ["0", "A", "B", "C"]


@dataclass
class DatasetItem:
    image_name: str
    image_path: Path
    label_path: Path
    counts: Dict[str, int]


def cls_to_group(cls_id: int) -> str:
    if cls_id == 0:
        return "0"
    if cls_id in (1, 2):
        return "A"
    if cls_id in (3, 4):
        return "B"
    if cls_id in (5, 6):
        return "C"
    return "UNK"


def parse_label_counts(label_path: Path) -> Dict[str, int]:
    out = {k: 0 for k in GROUPS}
    out["total"] = 0
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split()
            if not sp:
                continue
            try:
                c = int(float(sp[0]))
            except Exception:
                continue
            g = cls_to_group(c)
            if g in out:
                out[g] += 1
                out["total"] += 1
    return out


def load_all_items(source_images: Path, source_labels: Path) -> Dict[str, DatasetItem]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    items: Dict[str, DatasetItem] = {}

    for p in sorted(source_images.iterdir()):
        if not p.is_file() or p.suffix.lower() not in img_exts:
            continue
        lp = source_labels / f"{p.stem}.txt"
        if not lp.exists():
            continue
        items[p.name] = DatasetItem(
            image_name=p.name,
            image_path=p,
            label_path=lp,
            counts=parse_label_counts(lp),
        )
    if not items:
        raise RuntimeError("No image/label pairs found in source dataset")
    return items


def split_stats_from_root(dataset_root: Path) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for sp in SPLITS:
        cnt = {k: 0 for k in GROUPS}
        cnt["total"] = 0
        lbl_dir = dataset_root / sp / "labels"
        if not lbl_dir.exists():
            raise RuntimeError(f"Missing label dir: {lbl_dir}")
        for p in sorted(lbl_dir.glob("*.txt")):
            c = parse_label_counts(p)
            for k in cnt:
                cnt[k] += c[k]
        out[sp] = cnt
    tot = {k: 0 for k in GROUPS}
    tot["total"] = 0
    for sp in SPLITS:
        for k in tot:
            tot[k] += out[sp][k]
    out["total"] = tot
    return out


def baseline_stats_from_legacy_table() -> Dict[str, Dict[str, int]]:
    # User-provided baseline table (total = 4646), emphasizing balanced split-wise ratios.
    out = {
        "train": {"0": 1350, "A": 1546, "B": 420, "C": 158},
        "val": {"0": 224, "A": 266, "B": 66, "C": 25},
        "test": {"0": 232, "A": 264, "B": 74, "C": 21},
    }
    for sp in SPLITS:
        out[sp]["total"] = sum(out[sp][g] for g in GROUPS)

    tot = {k: 0 for k in GROUPS}
    tot["total"] = 0
    for sp in SPLITS:
        for k in tot:
            tot[k] += out[sp][k]
    out["total"] = tot
    return out


def compute_class_targets(
    baseline_stats: Dict[str, Dict[str, int]],
    new_totals: Dict[str, int],
) -> Dict[str, Dict[str, int]]:
    targets = {sp: {g: 0 for g in GROUPS} for sp in SPLITS}

    for g in GROUPS:
        old_total = baseline_stats["total"][g]
        if old_total <= 0:
            # If baseline has none for this class, split by baseline split-size shares.
            split_share = {
                sp: baseline_stats[sp]["total"] / max(1, baseline_stats["total"]["total"])
                for sp in SPLITS
            }
        else:
            split_share = {sp: baseline_stats[sp][g] / old_total for sp in SPLITS}

        raw = {sp: split_share[sp] * new_totals[g] for sp in SPLITS}
        flo = {sp: int(raw[sp]) for sp in SPLITS}
        rem = new_totals[g] - sum(flo.values())
        frac_order = sorted(SPLITS, key=lambda s: raw[s] - flo[s], reverse=True)
        for i in range(rem):
            flo[frac_order[i % len(SPLITS)]] += 1
        for sp in SPLITS:
            targets[sp][g] = flo[sp]

    return targets


def compute_total_targets(
    baseline_stats: Dict[str, Dict[str, int]],
    new_total_labels: int,
) -> Dict[str, int]:
    base_total = baseline_stats["total"]["total"]
    raw = {sp: baseline_stats[sp]["total"] / base_total * new_total_labels for sp in SPLITS}
    flo = {sp: int(raw[sp]) for sp in SPLITS}
    rem = new_total_labels - sum(flo.values())
    frac_order = sorted(SPLITS, key=lambda s: raw[s] - flo[s], reverse=True)
    for i in range(rem):
        flo[frac_order[i % len(SPLITS)]] += 1
    return flo


def compute_image_targets(
    baseline_stats: Dict[str, Dict[str, int]],
    n_images: int,
) -> Dict[str, int]:
    base_total = baseline_stats["total"]["total"]
    raw = {sp: baseline_stats[sp]["total"] / base_total * n_images for sp in SPLITS}
    flo = {sp: int(raw[sp]) for sp in SPLITS}
    rem = n_images - sum(flo.values())
    frac_order = sorted(SPLITS, key=lambda s: raw[s] - flo[s], reverse=True)
    for i in range(rem):
        flo[frac_order[i % len(SPLITS)]] += 1
    return flo


def read_initial_assignment(current_split_root: Path, image_names: List[str]) -> Dict[str, str]:
    assignment: Dict[str, str] = {}
    known = set(image_names)

    for sp in SPLITS:
        img_dir = current_split_root / sp / "images"
        if not img_dir.exists():
            raise RuntimeError(f"Missing split image dir: {img_dir}")
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.name in known:
                if p.name in assignment:
                    raise RuntimeError(f"Image appears in multiple splits: {p.name}")
                assignment[p.name] = sp

    missing = [n for n in image_names if n not in assignment]
    if missing:
        raise RuntimeError(f"Initial split missing {len(missing)} images")

    return assignment


def split_image_targets(assignment: Dict[str, str]) -> Dict[str, int]:
    out = {sp: 0 for sp in SPLITS}
    for sp in assignment.values():
        out[sp] += 1
    return out


def build_counts_for_assignment(
    assignment: Dict[str, str],
    items: Dict[str, DatasetItem],
) -> Dict[str, Dict[str, int]]:
    counts = {sp: {g: 0 for g in GROUPS} for sp in SPLITS}
    for sp in SPLITS:
        counts[sp]["total"] = 0

    for img_name, sp in assignment.items():
        c = items[img_name].counts
        for k in counts[sp]:
            counts[sp][k] += c[k]
    return counts


def split_contrib(
    split_count: Dict[str, int],
    class_target: Dict[str, int],
    total_target: int,
    class_weights: Dict[str, float],
    total_weight: float,
) -> float:
    err = 0.0
    for g in GROUPS:
        d = split_count[g] - class_target[g]
        err += class_weights[g] * (d * d) / max(1.0, float(class_target[g]))
    dt = split_count["total"] - total_target
    err += total_weight * (dt * dt) / max(1.0, float(total_target))
    return err


def objective(
    counts: Dict[str, Dict[str, int]],
    class_targets: Dict[str, Dict[str, int]],
    total_targets: Dict[str, int],
    class_weights: Dict[str, float],
    total_weight: float,
) -> float:
    return sum(
        split_contrib(counts[sp], class_targets[sp], total_targets[sp], class_weights, total_weight)
        for sp in SPLITS
    )


def apply_swap_update(
    counts: Dict[str, Dict[str, int]],
    a: str,
    b: str,
    va: Dict[str, int],
    vb: Dict[str, int],
) -> None:
    for k in counts[a]:
        counts[a][k] += vb[k] - va[k]
        counts[b][k] += va[k] - vb[k]


def optimize_by_swaps(
    base_assignment: Dict[str, str],
    items: Dict[str, DatasetItem],
    class_targets: Dict[str, Dict[str, int]],
    total_targets: Dict[str, int],
    class_weights: Dict[str, float],
    total_weight: float,
    max_iters: int,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]], float, int]:
    assignment = dict(base_assignment)
    split_to_imgs = {sp: [n for n, s in assignment.items() if s == sp] for sp in SPLITS}
    counts = build_counts_for_assignment(assignment, items)

    current_obj = objective(counts, class_targets, total_targets, class_weights, total_weight)
    swaps_done = 0

    for _ in range(max_iters):
        best_delta = 0.0
        best_move = None

        for i in range(len(SPLITS)):
            for j in range(i + 1, len(SPLITS)):
                a = SPLITS[i]
                b = SPLITS[j]

                old_local = split_contrib(counts[a], class_targets[a], total_targets[a], class_weights, total_weight)
                old_local += split_contrib(counts[b], class_targets[b], total_targets[b], class_weights, total_weight)

                for ia in split_to_imgs[a]:
                    va = items[ia].counts
                    for ib in split_to_imgs[b]:
                        vb = items[ib].counts

                        new_a = {k: counts[a][k] + vb[k] - va[k] for k in counts[a]}
                        new_b = {k: counts[b][k] + va[k] - vb[k] for k in counts[b]}

                        new_local = split_contrib(new_a, class_targets[a], total_targets[a], class_weights, total_weight)
                        new_local += split_contrib(new_b, class_targets[b], total_targets[b], class_weights, total_weight)

                        delta = new_local - old_local
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (a, b, ia, ib, va, vb)

        if best_move is None:
            break

        a, b, ia, ib, va, vb = best_move

        assignment[ia] = b
        assignment[ib] = a
        split_to_imgs[a].remove(ia)
        split_to_imgs[b].remove(ib)
        split_to_imgs[a].append(ib)
        split_to_imgs[b].append(ia)

        apply_swap_update(counts, a, b, va, vb)
        current_obj += best_delta
        swaps_done += 1

    return assignment, counts, current_obj, swaps_done


def random_perturb(
    assignment: Dict[str, str],
    rng: random.Random,
    num_swaps: int,
) -> Dict[str, str]:
    out = dict(assignment)
    split_to_imgs = {sp: [n for n, s in out.items() if s == sp] for sp in SPLITS}

    for _ in range(num_swaps):
        a, b = rng.sample(SPLITS, 2)
        if not split_to_imgs[a] or not split_to_imgs[b]:
            continue
        ia = rng.choice(split_to_imgs[a])
        ib = rng.choice(split_to_imgs[b])

        out[ia] = b
        out[ib] = a
        split_to_imgs[a].remove(ia)
        split_to_imgs[b].remove(ib)
        split_to_imgs[a].append(ib)
        split_to_imgs[b].append(ia)

    return out


def move_to_target_image_counts(
    assignment: Dict[str, str],
    items: Dict[str, DatasetItem],
    class_targets: Dict[str, Dict[str, int]],
    total_targets: Dict[str, int],
    class_weights: Dict[str, float],
    total_weight: float,
    image_targets: Dict[str, int],
) -> Dict[str, str]:
    out = dict(assignment)
    split_to_imgs = {sp: [n for n, s in out.items() if s == sp] for sp in SPLITS}
    counts = build_counts_for_assignment(out, items)

    while True:
        cur_img_counts = {sp: len(split_to_imgs[sp]) for sp in SPLITS}
        donors = [sp for sp in SPLITS if cur_img_counts[sp] > image_targets[sp]]
        receivers = [sp for sp in SPLITS if cur_img_counts[sp] < image_targets[sp]]
        if not donors and not receivers:
            break

        best = None
        best_delta = None
        for d in donors:
            for r in receivers:
                old_local = split_contrib(counts[d], class_targets[d], total_targets[d], class_weights, total_weight)
                old_local += split_contrib(counts[r], class_targets[r], total_targets[r], class_weights, total_weight)
                for img in split_to_imgs[d]:
                    v = items[img].counts
                    new_d = {k: counts[d][k] - v[k] for k in counts[d]}
                    new_r = {k: counts[r][k] + v[k] for k in counts[r]}
                    new_local = split_contrib(new_d, class_targets[d], total_targets[d], class_weights, total_weight)
                    new_local += split_contrib(new_r, class_targets[r], total_targets[r], class_weights, total_weight)
                    delta = new_local - old_local
                    if best is None or delta < best_delta:
                        best = (d, r, img, v)
                        best_delta = delta

        if best is None:
            break

        d, r, img, v = best
        out[img] = r
        split_to_imgs[d].remove(img)
        split_to_imgs[r].append(img)
        for k in counts[d]:
            counts[d][k] -= v[k]
            counts[r][k] += v[k]

    return out


def build_split_root(
    assignment: Dict[str, str],
    items: Dict[str, DatasetItem],
    out_root: Path,
) -> None:
    for sp in SPLITS:
        for sub in ["images", "labels"]:
            d = out_root / sp / sub
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

    for name, sp in assignment.items():
        item = items[name]
        shutil.copy2(item.image_path, out_root / sp / "images" / item.image_path.name)
        shutil.copy2(item.label_path, out_root / sp / "labels" / item.label_path.name)


def build_roi_csvs_from_assignment(
    assignment: Dict[str, str],
    items: Dict[str, DatasetItem],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {sp: [] for sp in SPLITS}
    cols_all = ["image_id", "gx", "gy", "gw", "gh", "icdas"]

    for img_name, sp in assignment.items():
        item = items[img_name]
        with Image.open(item.image_path) as im:
            w_img, h_img = im.size

        with item.label_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().split()
                if len(s) < 5:
                    continue
                cls = int(float(s[0]))
                x, y, w, h = map(float, s[1:5])
                gx = (x - w / 2.0) * w_img
                gy = (y - h / 2.0) * h_img
                gw = w * w_img
                gh = h * h_img
                rows_by_split[sp].append([img_name, gx, gy, gw, gh, cls])

    for sp in SPLITS:
        df_all = pd.DataFrame(rows_by_split[sp], columns=cols_all)
        df_pos = df_all[df_all["icdas"] >= 1].copy()
        df_all.to_csv(out_dir / f"icdas_strong_labels_all_{sp}.csv", index=False)
        df_pos.to_csv(out_dir / f"icdas_strong_labels_{sp}.csv", index=False)

    comb = pd.concat(
        [
            pd.DataFrame(rows_by_split["train"], columns=cols_all),
            pd.DataFrame(rows_by_split["val"], columns=cols_all),
            pd.DataFrame(rows_by_split["test"], columns=cols_all),
        ],
        ignore_index=True,
    )
    out_comb = pd.DataFrame(
        {
            "image_id": comb["image_id"],
            "tooth_id": "",
            "surface": "",
            "icdas": comb["icdas"],
            "gx": comb["gx"],
            "gy": comb["gy"],
            "gw": comb["gw"],
            "gh": comb["gh"],
        }
    )
    out_comb.to_csv(out_dir / "icdas_strong_labels.csv", index=False)

    def map_ic4(x: int) -> int:
        if x <= 0:
            return 0
        if x <= 2:
            return 1
        if x <= 4:
            return 2
        return 3

    for sp in SPLITS:
        src = pd.read_csv(out_dir / f"icdas_strong_labels_all_{sp}.csv")
        src["image_id"] = src["image_id"].astype(str).str.strip()
        src["ic4"] = src["icdas"].apply(map_ic4)
        src["x"] = src["gx"]
        src["y"] = src["gy"]
        src["w"] = src["gw"]
        src["h"] = src["gh"]
        src["y_ge1"] = (src["icdas"] >= 1).astype(int)
        src["y_ge2"] = 0
        src["y_ge3"] = (src["icdas"] >= 3).astype(int)
        src["y_ge4"] = 0
        src["y_ge5"] = (src["icdas"] >= 5).astype(int)
        src["y_ge6"] = 0
        src["mask_ge1"] = 1
        src["mask_ge2"] = 0
        src["mask_ge3"] = 1
        src["mask_ge4"] = 0
        src["mask_ge5"] = 1
        src["mask_ge6"] = 0

        cols = [
            "image_id",
            "x",
            "y",
            "w",
            "h",
            "gx",
            "gy",
            "gw",
            "gh",
            "icdas",
            "ic4",
            "y_ge1",
            "y_ge2",
            "y_ge3",
            "y_ge4",
            "y_ge5",
            "y_ge6",
            "mask_ge1",
            "mask_ge2",
            "mask_ge3",
            "mask_ge4",
            "mask_ge5",
            "mask_ge6",
        ]
        src[cols].to_csv(out_dir / f"icdas4_{sp}.csv", index=False)

    for sp in ["train", "val"]:
        d = pd.read_csv(out_dir / f"icdas4_{sp}.csv")
        roi = d[
            [
                "image_id",
                "x",
                "y",
                "w",
                "h",
                "icdas",
                "y_ge1",
                "y_ge2",
                "y_ge3",
                "y_ge4",
                "y_ge5",
                "y_ge6",
                "mask_ge1",
                "mask_ge2",
                "mask_ge3",
                "mask_ge4",
                "mask_ge5",
                "mask_ge6",
            ]
        ].copy()
        roi = roi.rename(columns={"icdas": "gt_icdas"})
        roi.to_csv(out_dir / f"roi_training_gt_strong.{sp}.csv", index=False)


def to_dataframe(stats: Dict[str, Dict[str, int]], tag: str) -> pd.DataFrame:
    rows = []
    for sp in SPLITS + ["total"]:
        total = stats[sp]["total"]
        for g in GROUPS:
            cnt = stats[sp][g]
            rows.append(
                {
                    "set": tag,
                    "split": sp,
                    "group": g,
                    "count": cnt,
                    "pct_in_split": 0.0 if total == 0 else cnt / total,
                    "split_total": total,
                }
            )
    return pd.DataFrame(rows)


def assignment_stats(
    assignment: Dict[str, str],
    items: Dict[str, DatasetItem],
) -> Dict[str, Dict[str, int]]:
    c = build_counts_for_assignment(assignment, items)
    tot = {k: 0 for k in GROUPS}
    tot["total"] = 0
    for sp in SPLITS:
        for k in tot:
            tot[k] += c[sp][k]
    c["total"] = tot
    return c


def image_split_counts(assignment: Dict[str, str]) -> Dict[str, int]:
    out = {sp: 0 for sp in SPLITS}
    for sp in assignment.values():
        out[sp] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebalance 309 split to better match old class/split proportions")
    parser.add_argument("--root", type=Path, default=Path("/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35"))
    parser.add_argument("--source-images", type=Path, default=None)
    parser.add_argument("--source-labels", type=Path, default=None)
    parser.add_argument(
        '--baseline-mode',
        type=str,
        default='legacy-table-4646',
        choices=['legacy-table-4646', 'dataset-root'],
        help='Baseline source: user historical table or labels under --baseline-split-root',
    )
    parser.add_argument(
        '--image-target-mode',
        type=str,
        default='baseline-ratio',
        choices=['baseline-ratio', 'current-locked'],
        help='Use baseline split ratio or keep current split image counts fixed.',
    )
    parser.add_argument("--baseline-split-root", type=Path, default=None)
    parser.add_argument("--current-split-root", type=Path, default=None)
    parser.add_argument("--out-split-root", type=Path, default=None)
    parser.add_argument("--out-derived-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--max-iters", type=int, default=120)
    parser.add_argument("--perturb-swaps", type=int, default=24)
    parser.add_argument("--report-name", type=str, default="ratio_match_309_vs_baseline.csv")
    parser.add_argument("--manifest-name", type=str, default="split_assignment_309_matched_oldratio.csv")
    args = parser.parse_args()

    root = args.root.resolve()
    source_images = args.source_images or (root / "datasets/309database/images")
    source_labels = args.source_labels or (root / "datasets/309database/labels")
    baseline_root = args.baseline_split_root or (root / "VOCdevkit_backup_20260421_224252")
    current_root = args.current_split_root or (root / "VOCdevkit")
    out_split_root = args.out_split_root or (root / "VOCdevkit_309_matched_oldratio")
    out_derived_dir = args.out_derived_dir or (root / "derived_309_matched_oldratio")

    rng = random.Random(args.seed)

    items = load_all_items(source_images, source_labels)
    image_names = sorted(items.keys())

    if args.baseline_mode == "legacy-table-4646":
        baseline_stats = baseline_stats_from_legacy_table()
    else:
        baseline_stats = split_stats_from_root(baseline_root)
    current_assignment = read_initial_assignment(current_root, image_names)
    current_stats = assignment_stats(current_assignment, items)

    if args.image_target_mode == 'baseline-ratio':
        image_targets = compute_image_targets(baseline_stats, len(image_names))
    else:
        image_targets = split_image_targets(current_assignment)

    new_totals = current_stats["total"]

    class_targets = compute_class_targets(baseline_stats, new_totals)
    total_targets = compute_total_targets(baseline_stats, new_totals["total"])

    class_weights = {"0": 1.0, "A": 1.0, "B": 2.2, "C": 4.0}
    total_weight = 0.8

    seeded_assignment = move_to_target_image_counts(
        current_assignment,
        items,
        class_targets,
        total_targets,
        class_weights,
        total_weight,
        image_targets,
    )
    seeded_stats = assignment_stats(seeded_assignment, items)

    best_assignment = dict(seeded_assignment)
    best_stats = seeded_stats
    best_obj = objective(best_stats, class_targets, total_targets, class_weights, total_weight)
    best_swaps = 0

    for r in range(args.restarts):
        if r == 0:
            init_assignment = dict(seeded_assignment)
        else:
            init_assignment = random_perturb(seeded_assignment, rng, args.perturb_swaps)

        # Keep split image counts unchanged from selected image target mode.
        if image_split_counts(init_assignment) != image_targets:
            raise RuntimeError("Perturbation changed image split targets, which should not happen")

        cand_assignment, _, cand_obj, swaps_done = optimize_by_swaps(
            init_assignment,
            items,
            class_targets,
            total_targets,
            class_weights,
            total_weight,
            args.max_iters,
        )
        cand_stats = assignment_stats(cand_assignment, items)

        if cand_obj < best_obj:
            best_obj = cand_obj
            best_assignment = cand_assignment
            best_stats = cand_stats
            best_swaps = swaps_done

    # Safety: keep split image counts equal to current split.
    if image_split_counts(best_assignment) != image_targets:
        raise RuntimeError("Best assignment violates image split count constraint")

    build_split_root(best_assignment, items, out_split_root)
    build_roi_csvs_from_assignment(best_assignment, items, out_derived_dir)

    baseline_df = to_dataframe(baseline_stats, "baseline")
    current_df = to_dataframe(current_stats, "current")
    matched_df = to_dataframe(best_stats, "matched")

    merged = (
        baseline_df.merge(current_df, on=["split", "group"], suffixes=("_baseline", "_current"))
        .merge(matched_df, on=["split", "group"], suffixes=("", "_matched"))
        .rename(
            columns={
                "count": "count_matched",
                "pct_in_split": "pct_in_split_matched",
                "split_total": "split_total_matched",
                "count_baseline": "count_baseline",
                "pct_in_split_baseline": "pct_in_split_baseline",
                "split_total_baseline": "split_total_baseline",
                "count_current": "count_current",
                "pct_in_split_current": "pct_in_split_current",
                "split_total_current": "split_total_current",
            }
        )
    )

    merged["delta_current_vs_baseline"] = merged["count_current"] - merged["count_baseline"]
    merged["delta_matched_vs_baseline"] = merged["count_matched"] - merged["count_baseline"]
    merged["delta_pp_current_vs_baseline"] = (
        merged["pct_in_split_current"] - merged["pct_in_split_baseline"]
    ) * 100.0
    merged["delta_pp_matched_vs_baseline"] = (
        merged["pct_in_split_matched"] - merged["pct_in_split_baseline"]
    ) * 100.0

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / args.report_name
    merged.to_csv(report_path, index=False)

    # Save image assignment manifest for reproducibility.
    manifest = pd.DataFrame(
        [{"image_id": name, "split": sp} for name, sp in sorted(best_assignment.items())]
    )
    manifest_path = report_dir / args.manifest_name
    manifest.to_csv(manifest_path, index=False)

    print("=== Done: rebalanced split generated ===")
    print(f"out_split_root={out_split_root}")
    print(f"out_derived_dir={out_derived_dir}")
    print(f"report_csv={report_path}")
    print(f"manifest_csv={manifest_path}")
    print(f"best_objective={best_obj:.6f}, best_swaps={best_swaps}")
    print(f"image_target_mode={args.image_target_mode}")
    print("image_count_targets:", image_targets)
    print("image_count_result(matched):", image_split_counts(best_assignment))

    print("\n=== Class targets (scaled from baseline class split shares) ===")
    for sp in SPLITS:
        print(sp, class_targets[sp])

    print("\n=== Current stats ===")
    for sp in SPLITS + ["total"]:
        c = current_stats[sp]
        print(
            f"{sp}: total={c['total']} 0={c['0']} A={c['A']} B={c['B']} C={c['C']}"
        )

    print("\n=== Matched stats ===")
    for sp in SPLITS + ["total"]:
        c = best_stats[sp]
        print(
            f"{sp}: total={c['total']} 0={c['0']} A={c['A']} B={c['B']} C={c['C']}"
        )


if __name__ == "__main__":
    main()
