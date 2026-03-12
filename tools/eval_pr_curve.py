# tools/eval_pr_curve.py

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torchvision.ops import nms

def norm_id(s):
    return os.path.basename(str(s)).strip()

def to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / ua

def do_nms(df, score_col, iou_thr):
    rows = []
    for img, g in df.groupby('image_id'):
        if len(g) == 0:
            continue
        boxes = torch.tensor(
            [to_xyxy(r['x'], r['y'], r['w'], r['h']) for _, r in g.iterrows()],
            dtype=torch.float32
        )
        scores = torch.tensor(g[score_col].values, dtype=torch.float32)
        keep = nms(boxes, scores, iou_thr).cpu().numpy().tolist()
        rows.append(g.iloc[keep])
    return pd.concat(rows, ignore_index=True) if rows else df.head(0)

def eval_ap_pr(dets, gt, iou_thr=0.5):
    """给定 NMS 之后的 dets（必须有列 score）和 gt，计算 AP@iou_thr 和 bestF1"""
    # 只评估阳性 GT
    if 'icdas' in gt.columns:
        gt = gt[gt['icdas'] >= 1].copy()
    n_gt = len(gt)
    if n_gt == 0:
        return 0.0, 0.0

    scores, tp, fp = [], [], []

    for img, gdet in dets.groupby('image_id'):
        ggt = gt[gt['image_id'] == img]
        if ggt.empty:
            # 这张图没有 GT，所有 det 都算 FP
            for _, dr in gdet.iterrows():
                scores.append(float(dr['score']))
                tp.append(0)
                fp.append(1)
            continue

        gboxes = [
            to_xyxy(r['gx'], r['gy'], r['gw'], r['gh'])
            for _, r in ggt.iterrows()
        ]
        used = [False] * len(gboxes)

        # 按 score 从高到低匹配
        gdet = gdet.sort_values('score', ascending=False)
        for _, dr in gdet.iterrows():
            dbox = to_xyxy(dr['x'], dr['y'], dr['w'], dr['h'])
            s = float(dr['score'])

            best_iou = 0.0
            best_j = -1
            for j, gbox in enumerate(gboxes):
                iou = iou_xyxy(dbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thr and best_j >= 0 and not used[best_j]:
                used[best_j] = True
                scores.append(s)
                tp.append(1)
                fp.append(0)
            else:
                scores.append(s)
                tp.append(0)
                fp.append(1)

    scores = np.array(scores)
    tp = np.array(tp)
    fp = np.array(fp)

    if len(scores) == 0:
        return 0.0, 0.0

    # 按分数全局排序
    order = np.argsort(-scores)
    tp = tp[order]
    fp = fp[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / float(n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # AP：连续版本（VOC-style）
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    # best F1
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_f1 = float(f1.max()) if len(f1) > 0 else 0.0

    return ap, best_f1

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)
    ap.add_argument('--gt_csv', required=True)
    ap.add_argument('--nms_iou', type=float, default=0.8)
    a = ap.parse_args()

    pred = pd.read_csv(a.pred_csv)
    gt   = pd.read_csv(a.gt_csv)

    # 统一 image_id
    for df in (pred, gt):
        df['image_id'] = df['image_id'].astype(str).apply(norm_id)

    # 只在重叠图像上评估
    imgs = sorted(set(pred['image_id']) & set(gt['image_id']))
    pred = pred[pred['image_id'].isin(imgs)].copy()
    gt   = gt[gt['image_id'].isin(imgs)].copy()

    # ===== BASE：用原始 YOLO 分 =====
    if 'yolo_score_raw' not in pred.columns:
        raise ValueError("pred_csv 里找不到列 'yolo_score_raw'，请确认列名。")

    base = pred.copy()
    base_nms = do_nms(base, 'yolo_score_raw', a.nms_iou)
    base_nms = base_nms.copy()
    base_nms['score'] = base_nms['yolo_score_raw']
    ap_base, f1_base = eval_ap_pr(base_nms, gt, iou_thr=0.5)
    print(f"[BASE]  AP@0.5={ap_base:.3f}  bestF1={f1_base:.3f}  (NMS IoU={a.nms_iou})")

    # ===== FUSED：自动寻找融合分列 =====
    fused = pred.copy()

    fused_score_col = None
    for cand in ['score_bump', 'score_fused_guard', 'score_gate', 'score_fused', 'yolo_score']:
        if cand in fused.columns:
            fused_score_col = cand
            break

    if fused_score_col is None:
        print("[FUSED] 找不到融合得分列（score_bump / score_fused_guard / score_fused / yolo_score），跳过 FUSED 评估。")
    else:
        fused_nms = do_nms(fused, fused_score_col, a.nms_iou)
        fused_nms = fused_nms.copy()
        fused_nms['score'] = fused_nms[fused_score_col]
        ap_fused, f1_fused = eval_ap_pr(fused_nms, gt, iou_thr=0.5)
        print(f"[FUSED] AP@0.5={ap_fused:.3f}  bestF1={f1_fused:.3f}  (NMS IoU={a.nms_iou}, score={fused_score_col})")
