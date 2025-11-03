import os, argparse, pandas as pd, numpy as np

def iou_xywh(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
    ix1,iy1=max(ax,bx),max(ay,by); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=aw*ah + bw*bh - inter + 1e-6
    return inter/ua

ap=argparse.ArgumentParser()
ap.add_argument('--pred_csv', required=True)
ap.add_argument('--gt_csv',   required=True)
ap.add_argument('--out_csv',  default='meta_train_val.csv')
ap.add_argument('--eval_iou', type=float, default=0.5)
a=ap.parse_args()

pred = pd.read_csv(a.pred_csv)
gt   = pd.read_csv(a.gt_csv).rename(columns={'gx':'x','gy':'y','gw':'w','gh':'h'})
pred['image_id']=pred['image_id'].apply(lambda s: os.path.basename(str(s)).strip())
gt['image_id']  =gt['image_id'].apply(lambda s: os.path.basename(str(s)).strip())
gt = gt[gt['icdas']>=1].copy()

rows=[]
for img,gp in pred.groupby('image_id'):
    ggt = gt[gt['image_id']==img]
    for _,r in gp.iterrows():
        hit=0
        for _,g in ggt.iterrows():
            if iou_xywh((r['x'],r['y'],r['w'],r['h']), (g['x'],g['y'],g['w'],g['h']))>=a.eval_iou:
                hit=1; break
        s = max(1e-6, min(1-1e-6, float(r['yolo_score_raw'] if 'yolo_score_raw' in r else r['yolo_score'])))
        p = max(1e-6, min(1-1e-6, float(r['p_ge1'])))
        # 特征：logit(s)、logit(p)、s、p、s*p、尺寸
        def logit(x): return np.log(x/(1-x))
        rows.append(dict(
            hit=hit, s=s, p=p, sp=s*p, logs=logit(s), logp=logit(p),
            w=float(r['w']), h=float(r['h']), area=float(r['w']*r['h'])
        ))
pd.DataFrame(rows).to_csv(a.out_csv, index=False)
print('wrote', a.out_csv, 'rows=', len(rows))
