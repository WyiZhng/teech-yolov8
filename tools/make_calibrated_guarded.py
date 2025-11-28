# tools/make_calibrated_guarded.py
import os, argparse, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression

def norm_id(s): return os.path.basename(str(s)).strip()
def to_xyxy(x,y,w,h): return [x, y, x+w, y+h]
def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua

ap=argparse.ArgumentParser()
ap.add_argument('--pred_csv', required=True)   # *_topK_scored.csv (需含 p_ge1, yolo_score_raw)
ap.add_argument('--gt_csv', required=True)     # icdas_strong_labels_val.csv (或 test)
ap.add_argument('--alpha', type=float, default=0.4)  # 线性融合权重
ap.add_argument('--tau',   type=float, default=0.6)  # 守门阈值：p_ge1_cal >= tau 才参与提升
ap.add_argument('--out_csv', required=True)
a=ap.parse_args()

pred = pd.read_csv(a.pred_csv)
for col in ['image_id','x','y','w','h','p_ge1','yolo_score_raw']:
    if col not in pred.columns:
        raise SystemExit(f"缺列 {col} 于 {a.pred_csv}")
pred['image_id'] = pred['image_id'].astype(str).apply(norm_id)

gt = pd.read_csv(a.gt_csv)
gt['image_id'] = gt['image_id'].astype(str).apply(norm_id)
if 'icdas' in gt.columns:
    gt = gt[gt['icdas']>=1].copy()

# 打命中标签 hit (IoU ≥ 0.5)
hits=[]
gt_by_img = {img:g.reset_index(drop=True) for img,g in gt.groupby('image_id')}
for i,r in pred.iterrows():
    img=r['image_id']
    if img not in gt_by_img: 
        hits.append(0); continue
    ok=False
    for _,gr in gt_by_img[img].iterrows():
        ok = iou(to_xyxy(r.x,r.y,r.w,r.h), to_xyxy(gr.gx,gr.gy,gr.gw,gr.gh)) >= 0.5
        if ok: break
    hits.append(int(ok))
pred['hit'] = hits

# Platt 校准 p_ge1 → p_ge1_cal
p = np.clip(pred['p_ge1'].values, 1e-6, 1-1e-6)
logit = np.log(p/(1-p)).reshape(-1,1)
lr = LogisticRegression().fit(logit, pred['hit'].values)
pred['p_ge1_cal'] = lr.predict_proba(logit)[:,1]

# Bump 分数：只对高置信候选加分
s   = np.clip(pred['yolo_score_raw'].values, 1e-9, 1.0)
p   = pred['p_ge1_cal'].values
tau = a.tau
lam = a.alpha

gain = np.clip(p - tau, 0.0, 1.0)
pred['score_bump'] = s * (1.0 + lam * gain)

pred.to_csv(a.out_csv, index=False)
print(f"wrote: {a.out_csv}  rows: {len(pred)}  alpha={a.alpha}  tau={a.tau}")
