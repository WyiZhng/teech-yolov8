import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def iou_xywh_norm(box1, box2):
    # box: [cx,cy,w,h] in normalized coords
    x1_c,y1_c,w1,h1 = box1
    x2_c,y2_c,w2,h2 = box2

    x1_min = x1_c - w1/2.
    x1_max = x1_c + w1/2.
    y1_min = y1_c - h1/2.
    y1_max = y1_c + h1/2.

    x2_min = x2_c - w2/2.
    x2_max = x2_c + w2/2.
    y2_min = y2_c - h2/2.
    y2_max = y2_c + h2/2.

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0., inter_xmax - inter_xmin)
    inter_h = max(0., inter_ymax - inter_ymin)
    inter = inter_w * inter_h
    area1 = w1*h1
    area2 = w2*h2
    union = area1 + area2 - inter + 1e-6
    return inter / union

def eval_grading(gt_csv, pred_csv, iou_thr=0.5, cls_col="icdas4"):
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)

    y_true = []
    y_pred = []

    # 先把 GT 的像素坐标转成归一化中心坐标（和 YOLO 一样）
    # 这里假设所有图像同尺寸，如果不确定，可以读一张图或在 GT CSV 里附宽高
    # 简化：假设宽 W=2000, 高 H=1300（如果不对你可以改成读图）
    W, H = 2000.0, 1300.0

    for img_id, g in gt_df.groupby("image_id"):
        # 找出该图所有预测
        p = pred_df[pred_df["image_id"] == img_id]
        
        # If no predictions for this image, we might want to count them as misses?
        # The current logic only counts "Matched GT". 
        # If a GT is not matched, it is ignored in this script's logic:
        # "If IoU < threshold: detection missed, ignore this GT"
        # This evaluates "Grading quality on recalled ROIs".
        
        if len(p) == 0:
            continue

        for _, r in g.iterrows():
            gx, gy, gw, gh = float(r["gx"]), float(r["gy"]), float(r["gw"]), float(r["gh"])
            gt_cls = int(r[cls_col])

            # Convert Top-Left (gx, gy) to Center (cx, cy)
            cx = gx + gw / 2.0
            cy = gy + gh / 2.0
            gt_box_norm = [cx/W, cy/H, gw/W, gh/H]

            # 对当前 GT 找 IoU 最大的预测
            best_iou = 0.0
            best_cls = None

            for _, pr in p.iterrows():
                pred_box = [pr["x_c"], pr["y_c"], pr["w_n"], pr["h_n"]]
                iou = iou_xywh_norm(gt_box_norm, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_cls = int(pr["cls_pred"])

            if best_iou >= iou_thr and best_cls is not None:
                y_true.append(gt_cls)
                y_pred.append(best_cls)
            # 如果 IoU < 阈值：说明检测没打中，这里可以忽略这条 GT；
            # 这样得到的是“在召回的 ROI 上的分级质量”，和你现在 ROI 头是同一个概念。

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("Matched GT count:", len(y_true))

    if len(y_true) > 0:
        mae = np.mean(np.abs(y_true - y_pred))
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

        print(f"MAE={mae:.3f}  QWK={qwk:.3f}")
        print("Confusion Matrix (rows=GT 0/A/B/C, cols=Pred 0/A/B/C):")
        print(cm)
    else:
        print("No matched GTs found.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_csv",   type=str, default="icdas4_val.csv")
    ap.add_argument("--pred_csv", type=str, default="pred_val_icdas4_yolo4cls.csv")
    ap.add_argument("--iou_thr",  type=float, default=0.5)
    ap.add_argument("--cls_col",  type=str, default="icdas4")
    args = ap.parse_args()

    eval_grading(args.gt_csv, args.pred_csv, args.iou_thr, args.cls_col)
