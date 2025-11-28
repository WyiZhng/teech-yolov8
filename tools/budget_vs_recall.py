# tools/budget_vs_recall.py  (fixed)
import os, argparse, pandas as pd, numpy as np

def norm_id(s): return os.path.basename(str(s)).strip()
def to_xyxy(x,y,w,h): return [x, y, x+w, y+h]
def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua

def recall_at_iou(pred, gt, thr=0.5):
    hits,total=0,len(gt)
    for img,g in gt.groupby('image_id'):
        cand = pred[pred['image_id']==img]
        if cand.empty: continue
        cxyxy = cand.apply(lambda r: to_xyxy(r['x'],r['y'],r['w'],r['h']), axis=1).to_list()
        for _,gr in g.iterrows():
            gxyxy = to_xyxy(gr['gx'],gr['gy'],gr['gw'],gr['gh'])
            ok = any(iou_xyxy(gxyxy, c) >= thr for c in cxyxy)
            hits += int(ok)
    return hits, total, hits/max(1,total)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)     # 你的 topK CSV
    ap.add_argument('--gt_csv', required=True)       # icdas_strong_labels_val.csv
    ap.add_argument('--use_score', required=True, help='列名：yolo_score 或 score_fused')
    ap.add_argument('--Ks', default='50,100,150,200,300,400')
    a=ap.parse_args()

    pred = pd.read_csv(a.pred_csv); pred['image_id']=pred['image_id'].apply(norm_id)
    gt   = pd.read_csv(a.gt_csv);   gt['image_id']=gt['image_id'].apply(norm_id)
    if 'icdas' in gt.columns: gt = gt[gt['icdas']>=1].copy()

    # 只在重叠图像上评估
    imgs = sorted(set(pred['image_id']) & set(gt['image_id']))
    pred = pred[pred['image_id'].isin(imgs)].copy()
    gt   = gt[gt['image_id'].isin(imgs)].copy()

    if a.use_score not in pred.columns:
        raise SystemExit(f"列 {a.use_score} 不存在。可用列：{list(pred.columns)}")

    print("Score列:", a.use_score)
    for K in map(int, a.Ks.split(',')):
        sub = (pred.sort_values(['image_id',a.use_score], ascending=[True,False])
                   .groupby('image_id').head(K).reset_index(drop=True))
        h,t,r = recall_at_iou(sub, gt, thr=0.5)
        print(f"K={K:>3}  Recall@0.5: {h}/{t}={r:.3f}")
