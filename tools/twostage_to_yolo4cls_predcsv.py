# tools/twostage_to_yolo4cls_predcsv.py
import argparse
import pandas as pd
import numpy as np

def xywh_to_xyxy(df, x='x', y='y', w='w', h='h'):
    x1 = df[x].values
    y1 = df[y].values
    x2 = x1 + df[w].values
    y2 = y1 + df[h].values
    return x1, y1, x2, y2

def nms_xyxy(x1, y1, x2, y2, scores, iou_thr=0.8):
    """Pure numpy NMS; returns kept indices (relative to input arrays)."""
    order = scores.argsort()[::-1]
    keep = []
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)

def infer_box_cols(df):
    cols = df.columns.tolist()
    # Try xywh
    if all(c in cols for c in ['x', 'y', 'w', 'h']):
        return 'xywh'
    # Some people use gx/gy/gw/gh
    if all(c in cols for c in ['gx', 'gy', 'gw', 'gh']):
        df.rename(columns={'gx':'x','gy':'y','gw':'w','gh':'h'}, inplace=True)
        return 'xywh'
    # Try xyxy
    if all(c in cols for c in ['x1', 'y1', 'x2', 'y2']):
        return 'xyxy'
    raise ValueError(f"Cannot find box columns in: {cols}")

def ordinal_to_ic4(p1, p3, p5, t1=0.5, t3=0.5, t5=0.5):
    # 0/A/B/C mapping
    # 0: p_ge1 < t1
    # A: p_ge1 >= t1 and p_ge3 < t3
    # B: p_ge3 >= t3 and p_ge5 < t5
    # C: p_ge5 >= t5
    ic4 = np.zeros_like(p1, dtype=np.int64)
    ic4[(p1 >= t1) & (p3 < t3)] = 1
    ic4[(p3 >= t3) & (p5 < t5)] = 2
    ic4[p5 >= t5] = 3
    return ic4

def softmax_to_ic4(df):
    # expects prob0, probA, probB, probC OR p_ge1, p_ge3, p_ge5
    if 'prob0' in df.columns:
        probs = df[['prob0','probA','probB','probC']].values
        return probs.argmax(axis=1).astype(np.int64)
    elif 'p_ge1' in df.columns:
        # Derive probs from p_ge
        p1 = df['p_ge1'].values
        p3 = df['p_ge3'].values
        p5 = df['p_ge5'].values
        
        # Clip and enforce monotonicity just in case
        p1 = np.clip(p1, 0, 1)
        p3 = np.clip(p3, 0, 1)
        p5 = np.clip(p5, 0, 1)
        p3 = np.minimum(p3, p1)
        p5 = np.minimum(p5, p3)
        
        prob0 = 1.0 - p1
        prob1 = p1 - p3
        prob2 = p3 - p5
        prob3 = p5
        
        probs = np.stack([prob0, prob1, prob2, prob3], axis=1)
        return probs.argmax(axis=1).astype(np.int64)
    else:
        raise ValueError("softmax mode requires prob0..C OR p_ge1..5 in CSV.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--mode', choices=['ordinal','softmax'], required=True)
    ap.add_argument('--score_col', default='yolo_score_raw', help='conf column used for NMS sorting')
    ap.add_argument('--nms_iou', type=float, default=0.80)
    ap.add_argument('--t1', type=float, default=0.5)
    ap.add_argument('--t3', type=float, default=0.5)
    ap.add_argument('--t5', type=float, default=0.5)
    ap.add_argument('--W', type=float, default=2000.0)
    ap.add_argument('--H', type=float, default=1300.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if 'image_id' not in df.columns:
        raise ValueError("in_csv must have column: image_id")

    box_type = infer_box_cols(df)
    if args.score_col not in df.columns:
        raise ValueError(f"score_col {args.score_col} not found in CSV columns: {df.columns.tolist()}")

    # make ic4
    if args.mode == 'ordinal':
        for c in ['p_ge1','p_ge3','p_ge5']:
            if c not in df.columns:
                raise ValueError(f"ordinal mode requires {c} in CSV.")
        df['ic4'] = ordinal_to_ic4(df['p_ge1'].values, df['p_ge3'].values, df['p_ge5'].values,
                                   t1=args.t1, t3=args.t3, t5=args.t5)
    else:
        # softmax mode
        df['ic4'] = softmax_to_ic4(df)

    # NMS per image
    outs = []
    for img, g in df.groupby('image_id'):
        g = g.copy()
        scores = g[args.score_col].values.astype(np.float32)

        if box_type == 'xywh':
            x1, y1, x2, y2 = xywh_to_xyxy(g, 'x','y','w','h')
        else:
            x1 = g['x1'].values; y1 = g['y1'].values
            x2 = g['x2'].values; y2 = g['y2'].values

        keep = nms_xyxy(x1, y1, x2, y2, scores, iou_thr=args.nms_iou)
        gk = g.iloc[keep].copy()

        # output columns: image_id,x,y,w,h,conf,ic4, cls_pred, x_c, y_c, w_n, h_n
        if box_type == 'xyxy':
            gk['x'] = gk['x1']; gk['y'] = gk['y1']
            gk['w'] = gk['x2'] - gk['x1']
            gk['h'] = gk['y2'] - gk['y1']
        
        # Normalize coordinates
        gk['x_c'] = (gk['x'] + gk['w'] / 2.0) / args.W
        gk['y_c'] = (gk['y'] + gk['h'] / 2.0) / args.H
        gk['w_n'] = gk['w'] / args.W
        gk['h_n'] = gk['h'] / args.H
        
        gk['conf'] = gk[args.score_col]
        gk['cls_pred'] = gk['ic4'] # Add cls_pred for compatibility
        
        outs.append(gk[['image_id','x','y','w','h','conf','ic4','cls_pred','x_c','y_c','w_n','h_n']])

    out = pd.concat(outs, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} rows={len(out)}  (mode={args.mode}, nms_iou={args.nms_iou}, score_col={args.score_col})")

if __name__ == '__main__':
    main()
