import os, argparse, pandas as pd, numpy as np

def iou_xywh(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
    ix1,iy1=max(ax,bx),max(ay,by); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=aw*ah+bw*bh-inter+1e-6
    return inter/ua

ap=argparse.ArgumentParser()
ap.add_argument('--prop_csv', required=True)     # train_predictions_topK.csv
ap.add_argument('--gt_csv', required=True)       # icdas_strong_labels_full_train.csv（含0）
ap.add_argument('--out_csv', required=True)
ap.add_argument('--pos_iou', type=float, default=0.5)
ap.add_argument('--neg_iou', type=float, default=0.3)
ap.add_argument('--max_negs_per_img', type=int, default=50)  # 控制负样本量
a=ap.parse_args()

prop = pd.read_csv(a.prop_csv); prop['image_id']=prop['image_id'].apply(os.path.basename)
gt   = pd.read_csv(a.gt_csv);   gt['image_id']=gt['image_id'].apply(os.path.basename)

rows=[]
for img, g in gt.groupby('image_id'):
    cand = prop[prop['image_id']==img].copy()
    # 正样本：逐个GT找匹配候选
    for _,gr in g.iterrows():
        icdas = int(gr['icdas'])
        if icdas>=1:
            best=None; best_iou=0
            for _,c in cand.iterrows():
                iou = iou_xywh((c['x'],c['y'],c['w'],c['h']), (gr['gx'],gr['gy'],gr['gw'],gr['gh']))
                if iou>best_iou: best_iou=iou; best=c
            if best is not None and best_iou>=a.pos_iou:
                rows.append([img,best['x'],best['y'],best['w'],best['h'],icdas])
    # 负样本：IoU<=neg_iou 的候选，优先分高的
    negs=[]
    for _,c in prop[prop['image_id']==img].iterrows():
        ious=[iou_xywh((c['x'],c['y'],c['w'],c['h']), (gr['gx'],gr['gy'],gr['gw'],gr['gh'])) for _,gr in g.iterrows()]
        miou = max(ious) if len(ious) else 0.0
        if miou<=a.neg_iou:
            negs.append((c['yolo_score'], c['x'],c['y'],c['w'],c['h']))
    negs=sorted(negs, key=lambda t:t[0], reverse=True)[:a.max_negs_per_img]
    for _,x,y,w,h in negs:
        rows.append([img,x,y,w,h,0])

df = pd.DataFrame(rows, columns=['image_id','x','y','w','h','gt_icdas'])
for k in range(1,7):
    df[f'y_ge{k}'] = (df['gt_icdas']>=k).astype(int)
    df[f'mask_ge{k}'] = 1
df.to_csv(a.out_csv, index=False)
print("wrote:", a.out_csv, "rows:", len(df))
