# tools/diag_pge1_on_proposals.py
import os, argparse, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def norm_id(s): return os.path.basename(str(s)).strip()
def to_xyxy(x,y,w,h): return [x, y, x+w, y+h]
def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua

ap=argparse.ArgumentParser()
ap.add_argument('--pred_csv', required=True)  # *_top400_scored.csv（含 p_ge1）
ap.add_argument('--gt_csv', required=True)    # icdas_strong_labels_val.csv（全 1）
a=ap.parse_args()

pred=pd.read_csv(a.pred_csv); pred['image_id']=pred['image_id'].apply(norm_id)
gt  =pd.read_csv(a.gt_csv);   gt['image_id']=gt['image_id'].apply(norm_id)
if 'icdas' in gt.columns: gt=gt[gt['icdas']>=1].copy()

# 命中标记
hits=[]
for img,g in gt.groupby('image_id'):
    cand=pred[pred['image_id']==img]
    if cand.empty: continue
    gxy=[to_xyxy(r.gx,r.gy,r.gw,r.gh) for _,r in g.iterrows()]
    for i,r in cand.iterrows():
        cxy=to_xyxy(r.x,r.y,r.w,r.h)
        ok=any(iou(cxy, t)>=0.5 for t in gxy)
        hits.append((i,int(ok)))
hits_idx, y = zip(*hits)
pred=pred.loc[list(hits_idx)].copy(); pred['hit']=y

scores=pred['p_ge1'].values; y=np.array(y)
auc=roc_auc_score(y,scores) if y.sum()>0 else float('nan')
ap =average_precision_score(y,scores) if y.sum()>0 else float('nan')

# Top-1@p_ge1（逐图取 p_ge1 最大的一个）
top=0; imgs=pred['image_id'].nunique()
for img,g in pred.groupby('image_id'):
    g=g.sort_values('p_ge1', ascending=False).head(1)
    top+=int(g['hit'].iloc[0]==1)
print(f"AUC={auc:.3f}  AP={ap:.3f}  Top1@p_ge1={top/max(1,imgs):.3f}  (imgs={imgs})")
