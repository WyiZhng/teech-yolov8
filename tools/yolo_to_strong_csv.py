# tools/yolo_to_strong_csv.py
import os, csv
from PIL import Image

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True, help='datasets/icdas_yolo 或 icdas_yolo_caries')
ap.add_argument('--out_prefix', default='icdas_strong_labels', help='输出前缀')
ap.add_argument('--keep_zero', action='store_true',
               help='是否保留 icdas=0（无龋）到CSV中。做proposal recall时可不保留；训练序位头建议保留。')
a = ap.parse_args()

# 你的新多类标注：0=无龋，1..6=龋齿等级
# 若用单类数据集（只保留了1..6），这段也兼容
CLASS_TO_ICDAS = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}

def dump_split(split):
    img_dir = os.path.join(a.root, split, 'images')
    lbl_dir = os.path.join(a.root, split, 'labels')
    rows = []
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith(('.jpg','.jpeg','.png')): 
            continue
        W,H = Image.open(os.path.join(img_dir, fn)).size
        txt = os.path.join(lbl_dir, os.path.splitext(fn)[0] + '.txt')
        if not os.path.exists(txt):
            continue  # 无目标图可不写入；训练序位头时也可写入一个带icdas=0的空行自行扩展
        for ln in open(txt):
            p = ln.strip().split()
            if len(p) != 5: 
                continue
            c, cx, cy, w, h = p
            c = int(float(c)); cx,cy,w,h = map(float, (cx,cy,w,h))
            # 归一化xywh -> 像素xywh（左上角坐标）
            gx = cx*W - w*W/2
            gy = cy*H - h*H/2
            gw = w*W
            gh = h*H
            icdas = CLASS_TO_ICDAS.get(c, 0)
            if not a.keep_zero and icdas == 0:
                continue
            rows.append([fn, gx, gy, gw, gh, icdas])
    out_csv = f"{a.out_prefix}_{split}.csv"
    with open(out_csv, 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['image_id','gx','gy','gw','gh','icdas'])
        cw.writerows(rows)
    print(f"Wrote {len(rows)} -> {out_csv}")

for sp in ['train','val','test']:
    dump_split(sp)
