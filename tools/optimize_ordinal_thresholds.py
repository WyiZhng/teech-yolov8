import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

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

def match_predictions(gt_df, pred_df, iou_thr=0.5, W=2000.0, H=1300.0):
    y_true = []
    p_ge1_list = []
    p_ge3_list = []
    p_ge5_list = []

    # Pre-calculate normalized coords for preds
    pred_df['x_c'] = (pred_df['x'] + pred_df['w'] / 2.0) / W
    pred_df['y_c'] = (pred_df['y'] + pred_df['h'] / 2.0) / H
    pred_df['w_n'] = pred_df['w'] / W
    pred_df['h_n'] = pred_df['h'] / H

    for img_id, g in gt_df.groupby("image_id"):
        p = pred_df[pred_df["image_id"] == img_id]
        if len(p) == 0:
            continue

        for _, r in g.iterrows():
            gx, gy, gw, gh = float(r["gx"]), float(r["gy"]), float(r["gw"]), float(r["gh"])
            gt_cls = int(r["ic4"]) # Assuming ic4 column exists in GT

            # Convert Top-Left (gx, gy) to Center (cx, cy)
            cx = gx + gw / 2.0
            cy = gy + gh / 2.0
            gt_box_norm = [cx/W, cy/H, gw/W, gh/H]

            best_iou = 0.0
            best_idx = -1

            # Vectorized IoU check could be faster but loop is fine for now
            for idx, pr in p.iterrows():
                pred_box = [pr["x_c"], pr["y_c"], pr["w_n"], pr["h_n"]]
                iou = iou_xywh_norm(gt_box_norm, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thr and best_idx != -1:
                row = p.loc[best_idx]
                y_true.append(gt_cls)
                p_ge1_list.append(row['p_ge1'])
                p_ge3_list.append(row['p_ge3'])
                p_ge5_list.append(row['p_ge5'])

    return np.array(y_true), np.array(p_ge1_list), np.array(p_ge3_list), np.array(p_ge5_list)

def get_preds(t, p1, p3, p5):
    t1, t3, t5 = t
    y_pred = np.zeros_like(p1, dtype=int)
    # Logic:
    # 3 if p5 >= t5
    # 2 if p3 >= t3 (and not 3)
    # 1 if p1 >= t1 (and not 2, 3)
    # 0 otherwise
    
    # Vectorized:
    # Start with 0
    # Set to 1 where p1 >= t1
    y_pred[p1 >= t1] = 1
    # Set to 2 where p3 >= t3 (overwrites 1)
    y_pred[p3 >= t3] = 2
    # Set to 3 where p5 >= t5 (overwrites 2)
    y_pred[p5 >= t5] = 3
    return y_pred

def objective(t, p1, p3, p5, y_true):
    y_pred = get_preds(t, p1, p3, p5)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return -qwk # Minimize negative QWK

def optimize_thresholds(gt_csv, pred_csv):
    print(f"Loading GT: {gt_csv}")
    gt_df = pd.read_csv(gt_csv)
    print(f"Loading Pred: {pred_csv}")
    pred_df = pd.read_csv(pred_csv)

    print("Matching predictions to GT...")
    y_true, p1, p3, p5 = match_predictions(gt_df, pred_df)
    print(f"Matched {len(y_true)} samples.")

    if len(y_true) == 0:
        print("No matches found.")
        return

    # Initial guess
    x0 = [0.5, 0.5, 0.5]
    
    # Bounds: 0.01 to 0.99
    bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]

    print("Optimizing thresholds...")
    # Nelder-Mead is robust for non-differentiable functions (like argmax/thresholding)
    # Powell is also good.
    res = minimize(objective, x0, args=(p1, p3, p5, y_true), method='Nelder-Mead', bounds=bounds) 
    # Note: Nelder-Mead doesn't strictly support bounds in scipy < 1.5 (approx), but L-BFGS-B requires gradients.
    # Let's try Powell or just simple grid search if this fails or gives weird results.
    # Actually, let's use a simple grid search first to get close, then fine tune?
    # Or just use Powell which is derivative-free.
    
    # Let's try a coarse grid search first to avoid local optima
    best_qwk = -1
    best_t = [0.5, 0.5, 0.5]
    
    for t1 in np.linspace(0.3, 0.8, 6):
        for t3 in np.linspace(0.3, 0.8, 6):
            for t5 in np.linspace(0.3, 0.8, 6):
                t = [t1, t3, t5]
                qwk = -objective(t, p1, p3, p5, y_true)
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_t = t
    
    print(f"Grid Search Best QWK: {best_qwk:.4f} at {best_t}")
    
    # Fine tune with Nelder-Mead starting from grid best
    res = minimize(objective, best_t, args=(p1, p3, p5, y_true), method='Nelder-Mead')
    
    final_t = res.x
    final_qwk = -res.fun
    
    print(f"Optimization Finished.")
    print(f"Best QWK: {final_qwk:.4f}")
    print(f"Optimal Thresholds: t1={final_t[0]:.4f}, t3={final_t[1]:.4f}, t5={final_t[2]:.4f}")
    
    # Calculate MAE
    y_pred = get_preds(final_t, p1, p3, p5)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"MAE at optimal thresholds: {mae:.4f}")

    return final_t

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--pred_csv", required=True)
    args = ap.parse_args()

    optimize_thresholds(args.gt_csv, args.pred_csv)
