#!/usr/bin/env python3
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


def main():
    root = Path('/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35')
    img_dir = root / 'datasets/309database/images'
    lbl_dir = root / 'datasets/309database/labels'
    seed = 20260421
    rng = random.Random(seed)

    old_split_files = {
        'train': root / 'icdas4_train.csv',
        'val': root / 'icdas4_val.csv',
        'test': root / 'icdas4_test.csv',
    }
    old_sets = {}
    for sp, p in old_split_files.items():
        df = pd.read_csv(p)
        old_sets[sp] = set(df['image_id'].astype(str).str.strip().tolist())

    if (old_sets['train'] & old_sets['val']) or (old_sets['train'] & old_sets['test']) or (old_sets['val'] & old_sets['test']):
        raise RuntimeError('Old split image sets overlap, aborting to avoid corruption.')

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_by_name = {}
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in img_exts:
            img_by_name[p.name] = p

    name_to_lbl = {}
    for img_name in img_by_name:
        stem = Path(img_name).stem
        lp = lbl_dir / f'{stem}.txt'
        if lp.exists():
            name_to_lbl[img_name] = lp

    all_images = sorted(name_to_lbl.keys())

    assign = {}
    for sp, sset in old_sets.items():
        for name in sset:
            if name in name_to_lbl:
                assign[name] = sp

    old_assigned_counts = {sp: sum(1 for v in assign.values() if v == sp) for sp in ['train', 'val', 'test']}
    old_total_baseline = sum(len(v) for v in old_sets.values())
    ratios = {sp: len(old_sets[sp]) / old_total_baseline for sp in ['train', 'val', 'test']}

    n_total = len(all_images)
    raw_targets = {sp: ratios[sp] * n_total for sp in ['train', 'val', 'test']}
    base_targets = {sp: int(raw_targets[sp]) for sp in ['train', 'val', 'test']}
    resid = n_total - sum(base_targets.values())
    frac_order = sorted(['train', 'val', 'test'], key=lambda s: raw_targets[s] - base_targets[s], reverse=True)
    for i in range(resid):
        base_targets[frac_order[i]] += 1

    deficits = {sp: base_targets[sp] - old_assigned_counts[sp] for sp in ['train', 'val', 'test']}
    if any(v < 0 for v in deficits.values()):
        for sp in ['train', 'val', 'test']:
            base_targets[sp] = max(base_targets[sp], old_assigned_counts[sp])
        extra = sum(base_targets.values()) - n_total
        if extra > 0:
            for sp in sorted(['train', 'val', 'test'], key=lambda s: base_targets[s], reverse=True):
                if extra <= 0:
                    break
                can = min(extra, base_targets[sp] - old_assigned_counts[sp])
                base_targets[sp] -= can
                extra -= can
        deficits = {sp: base_targets[sp] - old_assigned_counts[sp] for sp in ['train', 'val', 'test']}

    new_images = [n for n in all_images if n not in assign]
    if sum(deficits.values()) != len(new_images):
        left = len(new_images)
        deficits = {sp: max(0, deficits[sp]) for sp in ['train', 'val', 'test']}
        need = left - sum(deficits.values())
        if need > 0:
            order = sorted(['train', 'val', 'test'], key=lambda s: ratios[s], reverse=True)
            i = 0
            while need > 0:
                deficits[order[i % 3]] += 1
                need -= 1
                i += 1
        elif need < 0:
            need = -need
            for sp in sorted(['train', 'val', 'test'], key=lambda s: deficits[s], reverse=True):
                if need == 0:
                    break
                cut = min(deficits[sp], need)
                deficits[sp] -= cut
                need -= cut

    rng.shuffle(new_images)
    idx = 0
    for sp in ['train', 'val', 'test']:
        for _ in range(deficits[sp]):
            assign[new_images[idx]] = sp
            idx += 1

    if idx != len(new_images):
        for j in range(idx, len(new_images)):
            assign[new_images[j]] = 'train'

    final_counts = {sp: sum(1 for v in assign.values() if v == sp) for sp in ['train', 'val', 'test']}
    if sum(final_counts.values()) != n_total:
        raise RuntimeError('Final split count mismatch')

    voc309 = root / 'VOCdevkit_309'
    for sp in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            d = voc309 / sp / sub
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

    for img_name, sp in assign.items():
        src_img = img_by_name[img_name]
        src_lbl = name_to_lbl[img_name]
        shutil.copy2(src_img, voc309 / sp / 'images' / src_img.name)
        shutil.copy2(src_lbl, voc309 / sp / 'labels' / src_lbl.name)

    cols_all = ['image_id', 'gx', 'gy', 'gw', 'gh', 'icdas']
    rows_by_split = {'train': [], 'val': [], 'test': []}

    for img_name, sp in assign.items():
        ip = img_by_name[img_name]
        lp = name_to_lbl[img_name]
        with Image.open(ip) as im:
            w_img, h_img = im.size
        with open(lp, 'r', encoding='utf-8') as f:
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

    for sp in ['train', 'val', 'test']:
        df_all = pd.DataFrame(rows_by_split[sp], columns=cols_all)
        df_pos = df_all[df_all['icdas'] >= 1].copy()
        df_all.to_csv(root / f'icdas_strong_labels_all_{sp}.csv', index=False)
        df_pos.to_csv(root / f'icdas_strong_labels_{sp}.csv', index=False)

    comb = pd.concat(
        [
            pd.DataFrame(rows_by_split['train'], columns=cols_all),
            pd.DataFrame(rows_by_split['val'], columns=cols_all),
            pd.DataFrame(rows_by_split['test'], columns=cols_all),
        ],
        ignore_index=True,
    )
    out_comb = pd.DataFrame(
        {
            'image_id': comb['image_id'],
            'tooth_id': '',
            'surface': '',
            'icdas': comb['icdas'],
            'gx': comb['gx'],
            'gy': comb['gy'],
            'gw': comb['gw'],
            'gh': comb['gh'],
        }
    )
    out_comb.to_csv(root / 'icdas_strong_labels.csv', index=False)

    def map_ic4(x):
        if x <= 0:
            return 0
        if x <= 2:
            return 1
        if x <= 4:
            return 2
        return 3

    for sp in ['train', 'val', 'test']:
        src = pd.read_csv(root / f'icdas_strong_labels_all_{sp}.csv')
        src['image_id'] = src['image_id'].astype(str).str.strip()
        src['ic4'] = src['icdas'].apply(map_ic4)
        src['x'] = src['gx']
        src['y'] = src['gy']
        src['w'] = src['gw']
        src['h'] = src['gh']
        src['y_ge1'] = (src['icdas'] >= 1).astype(int)
        src['y_ge2'] = 0
        src['y_ge3'] = (src['icdas'] >= 3).astype(int)
        src['y_ge4'] = 0
        src['y_ge5'] = (src['icdas'] >= 5).astype(int)
        src['y_ge6'] = 0
        src['mask_ge1'] = 1
        src['mask_ge2'] = 0
        src['mask_ge3'] = 1
        src['mask_ge4'] = 0
        src['mask_ge5'] = 1
        src['mask_ge6'] = 0
        cols = [
            'image_id',
            'x',
            'y',
            'w',
            'h',
            'gx',
            'gy',
            'gw',
            'gh',
            'icdas',
            'ic4',
            'y_ge1',
            'y_ge2',
            'y_ge3',
            'y_ge4',
            'y_ge5',
            'y_ge6',
            'mask_ge1',
            'mask_ge2',
            'mask_ge3',
            'mask_ge4',
            'mask_ge5',
            'mask_ge6',
        ]
        src[cols].to_csv(root / f'icdas4_{sp}.csv', index=False)

    for sp in ['train', 'val']:
        d = pd.read_csv(root / f'icdas4_{sp}.csv')
        roi = d[
            [
                'image_id',
                'x',
                'y',
                'w',
                'h',
                'icdas',
                'y_ge1',
                'y_ge2',
                'y_ge3',
                'y_ge4',
                'y_ge5',
                'y_ge6',
                'mask_ge1',
                'mask_ge2',
                'mask_ge3',
                'mask_ge4',
                'mask_ge5',
                'mask_ge6',
            ]
        ].copy()
        roi = roi.rename(columns={'icdas': 'gt_icdas'})
        roi.to_csv(root / f'roi_training_gt_strong.{sp}.csv', index=False)

    for sp in ['train', 'val']:
        roi = pd.read_csv(root / f'roi_training_gt_strong.{sp}.csv')
        exist_dir = voc309 / sp / 'images'
        keep = roi['image_id'].apply(lambda x: (exist_dir / str(x)).exists())
        roi[keep].to_csv(root / f'roi_training_gt_strong.{sp}.in_root.csv', index=False)

    summary = {}
    for sp in ['train', 'val', 'test']:
        d_all = pd.read_csv(root / f'icdas_strong_labels_all_{sp}.csv')
        summary[sp] = {
            'images': int(sum(1 for _, s in assign.items() if s == sp)),
            'rois_all': int(len(d_all)),
            'rois_pos': int((d_all['icdas'] >= 1).sum()),
            'class_counts_0_6': {int(k): int(v) for k, v in d_all['icdas'].value_counts().sort_index().items()},
        }

    print('Split image counts:', final_counts)
    print('Targets used:', base_targets)
    print('New images assigned:', len(new_images))
    print('VOC split root:', voc309)
    print('Summary:', summary)


if __name__ == '__main__':
    main()
