
import argparse, os
import pandas as pd
import numpy as np

def normalize_image_id_col(df, col='image_id', make_basename=True):
    if col not in df.columns:
        return df
    df[col] = df[col].astype(str).str.strip()
    if make_basename:
        df[col] = df[col].apply(lambda s: os.path.basename(s))
    return df

def load_and_standardize_predictions(path, assume_has_header=True):
    # Return columns: image_id,x,y,w,h,yolo_score (numeric)
    if assume_has_header:
        try:
            df = pd.read_csv(path)
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            rename_map = {}
            for a,b in [('image','image_id'),('file','image_id'),('name','image_id'),
                        ('score','yolo_score'),('conf','yolo_score'),
                        ('cx','x'),('cy','y'),('width','w'),('height','h'),
                        ('xcenter','x'),('ycenter','y'),('x_center','x'),('y_center','y')]:
                if a in df.columns and b not in df.columns:
                    rename_map[a] = b
            df = df.rename(columns=rename_map)
            expected = ['image_id','x','y','w','h','yolo_score']
            if all(c in df.columns for c in expected):
                df = df[expected].copy()
                for c in ['x','y','w','h','yolo_score']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.dropna(subset=['image_id','x','y','w','h','yolo_score'])
                return df
        except Exception:
            pass
    # Fallback header=None
    df = pd.read_csv(path, header=None)
    # best effort: assume first 6 cols match
    if df.shape[1] < 6:
        raise ValueError(f"Cannot parse predictions CSV: {path}")
    df = df.iloc[:, :6].copy()
    df.columns = ['image_id','x','y','w','h','yolo_score']
    for c in ['x','y','w','h','yolo_score']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['image_id','x','y','w','h','yolo_score'])
    return df

def topk_per_image(df, k=300):
    return (df.sort_values(['image_id','yolo_score'], ascending=[True, False])
              .groupby('image_id', as_index=False).head(k)
              .reset_index(drop=True))

def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h

def iou_pair(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter / ua

def greedy_match(image_df, gt_df, iou_thr=0.5):
    # Prepare output with scalar columns only
    out = image_df.copy()
    out['matched'] = False
    out['gt_icdas'] = np.nan
    out['iou'] = 0.0
    out['gt_gx'] = np.nan
    out['gt_gy'] = np.nan
    out['gt_gw'] = np.nan
    out['gt_gh'] = np.nan

    if gt_df.empty or image_df.empty:
        return out

    # boxes and order
    boxes = [xywh_to_xyxy(*r[1][['x','y','w','h']]) for r in image_df.iterrows()]
    order = np.argsort(-image_df['yolo_score'].values)
    gt_df = gt_df.reset_index(drop=True)
    taken = set()

    for ci in order:
        best, best_iou = None, -1.0
        for gi, gr in gt_df.iterrows():
            if gi in taken:
                continue
            ov = iou_pair(boxes[ci], xywh_to_xyxy(float(gr['gx']), float(gr['gy']), float(gr['gw']), float(gr['gh'])))
            if ov >= iou_thr and ov > best_iou:
                best, best_iou = gi, ov
        if best is not None:
            taken.add(best)
            out.at[out.index[ci], 'matched'] = True
            out.at[out.index[ci], 'gt_icdas'] = int(gt_df.at[best, 'icdas'])
            out.at[out.index[ci], 'iou'] = float(best_iou)
            out.at[out.index[ci], 'gt_gx'] = float(gt_df.at[best, 'gx'])
            out.at[out.index[ci], 'gt_gy'] = float(gt_df.at[best, 'gy'])
            out.at[out.index[ci], 'gt_gw'] = float(gt_df.at[best, 'gw'])
            out.at[out.index[ci], 'gt_gh'] = float(gt_df.at[best, 'gh'])
    return out

def add_ordinal_targets(df):
    for k in range(1,7):
        coly, colm = f'y_ge{k}', f'mask_ge{k}'
        df[coly] = np.nan
        df[colm] = 0
    strong = df['gt_icdas'].notna()
    if strong.any():
        g = df.loc[strong, 'gt_icdas'].astype(int)
        for k in range(1,7):
            df.loc[strong, f'y_ge{k}'] = (g >= k).astype(int)
            df.loc[strong, f'mask_ge{k}'] = 1
    # explicit strong negatives
    neg = df['gt_icdas'].fillna(-1).astype(int) == 0
    if neg.any():
        for k in range(1,7):
            df.loc[neg, f'y_ge{k}'] = 0
            df.loc[neg, f'mask_ge{k}'] = 1
    return df

def align_safe(cand_df, gt_df, iou_thr=0.5):
    # supervise only images that exist in GT
    gt_imgs = set(gt_df['image_id'])
    out_rows = []
    pos_by_img = gt_df.groupby('image_id')['icdas'].apply(lambda s: (s>=1).any()).to_dict()
    for img, df_img in cand_df.groupby('image_id'):
        if img in gt_imgs:
            gt_img = gt_df[gt_df['image_id']==img]
            matched = greedy_match(df_img.reset_index(drop=True), gt_img.reset_index(drop=True), iou_thr=iou_thr)
            # if this image has no positives in GT, set unmatched to strong negatives
            if not pos_by_img.get(img, False):
                um = matched['matched'] == False
                matched.loc[um, 'gt_icdas'] = 0
        else:
            matched = df_img.copy()
            matched['matched'] = False
            matched['gt_icdas'] = np.nan
            matched['iou'] = 0.0
            for c in ['gt_gx','gt_gy','gt_gw','gt_gh']:
                matched[c] = np.nan
        out_rows.append(matched)
    out = pd.concat(out_rows, ignore_index=True)
    out = add_ordinal_targets(out)
    return out

def main(args):
    # Load predictions
    train_std = load_and_standardize_predictions(args.train_preds, assume_has_header=True)
    val_std   = load_and_standardize_predictions(args.val_preds,   assume_has_header=True)
    train_std = normalize_image_id_col(train_std, 'image_id', make_basename=not args.keep_full_path)
    val_std   = normalize_image_id_col(val_std,   'image_id', make_basename=not args.keep_full_path)
    train_std = topk_per_image(train_std, args.topk)
    val_std   = topk_per_image(val_std,   args.topk)

    # Load GT strong labels
    gt = pd.read_csv(args.strong_labels)
    gt.columns = [str(c).strip().lower() for c in gt.columns]
    required = ['image_id','icdas','gx','gy','gw','gh']
    miss = [c for c in required if c not in gt.columns]
    if miss:
        raise ValueError(f"Missing columns in strong labels: {miss}. Required: {required}")
    gt = normalize_image_id_col(gt, 'image_id', make_basename=not args.keep_full_path)

    # Align
    aligned_train = align_safe(train_std, gt, iou_thr=args.iou)
    aligned_val   = align_safe(val_std,   gt, iou_thr=args.iou)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    train_out = os.path.join(args.out_dir, 'roi_training_strong.train.csv')
    val_out   = os.path.join(args.out_dir, 'roi_training_strong.val.csv')
    aligned_train.to_csv(train_out, index=False)
    aligned_val.to_csv(val_out, index=False)

    def summarize(df):
        total = len(df)
        matched = int(df['matched'].sum())
        pos = int((df['gt_icdas'].fillna(-1) >= 1).sum())
        neg = int((df['gt_icdas'].fillna(-1) == 0).sum())
        ign = total - pos - neg
        return {'total': total, 'matched': matched, 'pos': pos, 'neg': neg, 'ignored': ign}
    print('train:', summarize(aligned_train))
    print('val:  ', summarize(aligned_val))
    print('Saved:\n ', train_out, '\n ', val_out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_preds', type=str, required=True)
    ap.add_argument('--val_preds', type=str, required=True)
    ap.add_argument('--strong_labels', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--topk', type=int, default=300)
    ap.add_argument('--keep_full_path', action='store_true',
                    help='Do not basename() image_id; use if your image_id in GT includes full paths.')
    args = ap.parse_args()
    main(args)
