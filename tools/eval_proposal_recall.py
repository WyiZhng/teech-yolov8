# tools/eval_proposal_recall.py
import os, argparse, pandas as pd
from PIL import Image
import numpy as np

def norm_id(s): return os.path.basename(str(s)).strip()

def to_xyxy(x,y,w,h): return [x, y, x+w, y+h]

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter + 1e-6
    return inter/ua

def maybe_denorm_xywh(df, img_root):
    """若w/h的95分位<=2，视为归一化到[0,1]，按图像像素还原"""
    if df.empty: return df
    wh95 = max(df['w'].quantile(0.95), df['h'].quantile(0.95))
    if wh95 > 2.0:  # 已是像素
        return df
    size_cache = {}
    xs,ys,ws,hs=[],[],[],[]
    for _,r in df.iterrows():
        im = r['image_id']
        if im not in size_cache:
            W,H = Image.open(os.path.join(img_root, im)).size
            size_cache[im]=(W,H)
        W,H=size_cache[im]
        xs.append(float(r['x'])*W); ys.append(float(r['y'])*H)
        ws.append(float(r['w'])*W); hs.append(float(r['h'])*H)
    df=df.copy()
    df['x'],df['y'],df['w'],df['h']=xs,ys,ws,hs
    return df

def recall_at_iou(pred, gt, thr=0.5):
    hits=0; total=len(gt)
    for img, gimg in gt.groupby('image_id'):
        cand = pred[pred['image_id']==img]
        if cand.empty: continue
        cxyxy = cand.apply(lambda r: to_xyxy(r['x'],r['y'],r['w'],r['h']), axis=1).to_list()
        for _,gr in gimg.iterrows():
            gxyxy = to_xyxy(gr['gx'],gr['gy'],gr['gw'],gr['gh'])
            ok = any(iou_xyxy(gxyxy, c) >= thr for c in cxyxy)
            hits += int(ok)
    return hits, total, (hits/max(1,total))

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)
    ap.add_argument('--gt_csv', required=True)
    ap.add_argument('--img_root', default=None, help='若pred/gt是归一化坐标，需要提供images根目录做像素还原')
    ap.add_argument('--topk', type=int, default=None, help='每图按yolo_score取前K个候选后再评估')
    a = ap.parse_args()

    pred = pd.read_csv(a.pred_csv)
    gt   = pd.read_csv(a.gt_csv)

    # 统一文件名
    for df in (pred, gt):
        df['image_id'] = df['image_id'].astype(str).apply(norm_id)

    # 只评估阳性GT（icdas>=1）；若没有该列则认为全为阳性
    if 'icdas' in gt.columns:
        gt = gt[gt['icdas']>=1].copy()

    # 坐标自适配（若提供img_root且疑似归一化）
    if a.img_root is not None:
        pred = maybe_denorm_xywh(pred, os.path.join(a.img_root))
        if set(['gx','gy','gw','gh']).issubset(gt.columns):
            # gt本就像素，无需处理；若你也存的是归一化，可仿照pred处理
            pass

    # 可选：每图裁前K个
    if a.topk is not None and 'yolo_score' in pred.columns:
        pred = pred.sort_values(['image_id','yolo_score'], ascending=[True,False]) \
                   .groupby('image_id').head(a.topk).reset_index(drop=True)

    # 只在两者重叠的图像上评估
    imgs = sorted(set(pred['image_id']) & set(gt['image_id']))
    pred = pred[pred['image_id'].isin(imgs)].copy()
    gt   = gt[gt['image_id'].isin(imgs)].copy()

    # 评估 R@0.5 和 R@0.3
    h5,t5,r5 = recall_at_iou(pred, gt, thr=0.5)
    h3,t3,r3 = recall_at_iou(pred, gt, thr=0.3)

    # 报告
    n_img = gt['image_id'].nunique()
    avg_cand = len(pred)/max(1,n_img)
    print(f"[Sanity] images: {n_img}  candidates: {len(pred)}  avg/image: {avg_cand:.1f}")
    print(f"Proposal Recall@0.5: {h5}/{t5} = {r5:.3f}")
    print(f"Proposal Recall@0.3: {h3}/{t3} = {r3:.3f}")
