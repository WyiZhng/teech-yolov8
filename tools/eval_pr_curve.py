import os, argparse, pandas as pd, numpy as np
from torchvision.ops import nms
import torch

def norm_id(s): return os.path.basename(str(s)).strip()
def to_xyxy(r): return [r['x'],r['y'],r['x']+r['w'],r['y']+r['h']]
def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua

def do_nms(df, score_col, iou_thr):
    rows=[]
    for img,g in df.groupby('image_id'):
        if len(g)==0: continue
        boxes  = torch.tensor([to_xyxy(r) for _,r in g.iterrows()], dtype=torch.float32)
        scores = torch.tensor(g[score_col].values, dtype=torch.float32)
        keep   = nms(boxes, scores, iou_threshold=iou_thr).cpu().numpy().tolist()
        rows.append(g.iloc[keep])
    return pd.concat(rows, ignore_index=True) if rows else df.head(0)

def pr_ap(df, gt, iou_thr=0.5):
    # 展开所有候选（按分数降序）
    D = df.sort_values('score', ascending=False).reset_index(drop=True)
    tp,fp=[],[]
    # 构建 per-image 的 GT 标记是否已匹配
    matched = {img: np.zeros(len(g), dtype=bool) for img,g in gt.groupby('image_id')}
    gt_by_img = {img: g.reset_index(drop=True) for img,g in gt.groupby('image_id')}
    for _,r in D.iterrows():
        img = r['image_id']
        if img not in gt_by_img:
            fp.append(1); tp.append(0); continue
        g = gt_by_img[img]
        # 找到与该候选 IoU 最大的 GT
        ious = np.array([iou_xyxy([r['x'],r['y'],r['x']+r['w'],r['y']+r['h']],
                                  [gr['gx'],gr['gy'],gr['gx']+gr['gw'],gr['gy']+gr['gh']]) for _,gr in g.iterrows()])
        if len(ious)==0 or ious.max()<iou_thr:
            fp.append(1); tp.append(0)
        else:
            j = ious.argmax()
            if matched[img][j]:  # 该GT已被别的候选匹配过 -> 重复命中算FP
                fp.append(1); tp.append(0)
            else:
                matched[img][j]=True
                tp.append(1); fp.append(0)
    tp=np.cumsum(tp); fp=np.cumsum(fp)
    rec = tp / max(1, sum(len(x) for x in matched.values()))
    prec= tp / np.maximum(1, tp+fp)
    # 11-point AP 或插值AP都可，这里用梯形近似
    ap = np.trapz(prec, rec)
    f1 = np.max(2*prec*rec/np.maximum(1e-9,prec+rec))
    return ap, f1, prec[-1] if len(prec)>0 else 0.0, rec[-1] if len(rec)>0 else 0.0

ap=argparse.ArgumentParser()
ap.add_argument('--pred_csv', required=True)   # 你的 top400 CSV，须包含 yolo_score_raw 与 score_fused
ap.add_argument('--gt_csv', required=True)     # icdas_strong_labels_val.csv
ap.add_argument('--nms_iou', type=float, default=0.80)
a=ap.parse_args()

pred = pd.read_csv(a.pred_csv); pred['image_id']=pred['image_id'].apply(norm_id)
gt   = pd.read_csv(a.gt_csv);   gt['image_id']=gt['image_id'].apply(norm_id)
if 'icdas' in gt.columns: gt=gt[gt['icdas']>=1].copy()
imgs = sorted(set(pred['image_id']) & set(gt['image_id']))
pred = pred[pred['image_id'].isin(imgs)].copy()
gt   = gt[gt['image_id'].isin(imgs)].copy()

for col,name in [('yolo_score_raw','BASE'), ('score_fused','FUSED')]:
    df = pred.rename(columns={col:'score'})
    df = do_nms(df, 'score', a.nms_iou)
    ap, f1, p_end, r_end = pr_ap(df, gt, iou_thr=0.5)
    print(f"[{name}]  AP@0.5={ap:.3f}  bestF1={f1:.3f}  (NMS IoU={a.nms_iou})")
