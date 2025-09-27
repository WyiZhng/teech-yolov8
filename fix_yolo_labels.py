#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os

from PIL import Image

DATA_ROOT = "/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/Benchmarking Dataset"
SUBSETS = ["train", "valid", "test"]

def fix_label_file(lbl_path, img_w, img_h):
    changed = False
    out_lines = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip().split()
            if not s:
                continue
            try:
                cls = int(float(s[0]))
            except:  # noqa: E722 - 容忍异常，默认类别为 0
                cls = 0
                s = ["0"] + s

            nums = list(map(float, s[1:]))
            # 分割：偶数个点 >= 6
            if len(nums) >= 6 and len(nums) % 2 == 0:
                xs = nums[0::2]
                ys = nums[1::2]
                if max(xs + [0]) > 1.0 or max(ys + [0]) > 1.0:
                    xs = [max(0.0, min(1.0, x / img_w)) for x in xs]
                    ys = [max(0.0, min(1.0, y / img_h)) for y in ys]
                    changed = True
                nums = [v for xy in zip(xs, ys) for v in xy]
                out_lines.append(" ".join([str(cls)] + [f"{v:.6f}" for v in nums]))
                continue

            if len(nums) < 4:
                continue
            coords = nums[:4]
            x1 = y1 = x2 = y2 = None

            def in01(a):
                return 0.0 <= a <= 1.0

            if all(in01(v) for v in coords):
                x, y, w, h = coords
            else:
                c0, c1, c2, c3 = coords
                if (c2 > c0 and c3 > c1) and (c2 > 1.0 or c3 > 1.0):
                    x1, y1, x2, y2 = c0, c1, c2, c3
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    x = x1 + w / 2.0
                    y = y1 + h / 2.0
                else:
                    x, y, w, h = coords

                x /= img_w
                w /= img_w
                y /= img_h
                h /= img_h
                changed = True

            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            if w == 0 or h == 0:
                continue

            out_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    if changed:
        bak = lbl_path + ".bak"
        if not os.path.exists(bak):
            os.rename(lbl_path, bak)
        with open(lbl_path, "w", encoding="utf-8") as f:
            for l in out_lines:
                f.write(l + "\n")
    return changed


def main():
    fixed_cnt = 0
    for split in SUBSETS:
        img_dir = os.path.join(DATA_ROOT, split, "images")
        lbl_dir = os.path.join(DATA_ROOT, split, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue
        for img_path in glob.glob(os.path.join(img_dir, "*.*")):
            name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, name + ".txt")
            if not os.path.exists(lbl_path):
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:  # noqa: BLE001
                print("Image open failed:", img_path, e)
                continue
            try:
                changed = fix_label_file(lbl_path, w, h)
                if changed:
                    fixed_cnt += 1
                    print("[fixed]", lbl_path)
            except Exception as e:  # noqa: BLE001
                print("Label fix failed:", lbl_path, e)
    print("Done. files fixed:", fixed_cnt)


if __name__ == "__main__":
    main()
