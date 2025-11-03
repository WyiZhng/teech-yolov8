# fuse_infer_and_eval_v2.py
import os
import argparse
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.ops import nms

def norm_id(s):  # basename 对齐
    return os.path.basename(str(s)).strip()

def to_xyxy_row(r):
    # 以候选框中心为基准做评估时的可选扩张（只影响评估 IoU，不影响裁剪送入分级器）
    cx = r['x'] + r['w'] * 0.5
    cy = r['y'] + r['h'] * 0.5
    w  = r['w'] * a.cand_expand
    h  = r['h'] * a.cand_expand
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return [x1, y1, x2, y2]


def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua

class OrdinalHead(torch.nn.Module):
    def __init__(self, out_dims=6):
        super().__init__()
        m = models.resnet18(weights=None); feat = m.fc.in_features; m.fc = torch.nn.Identity()
        self.backbone = m; self.head = torch.nn.Linear(feat, out_dims)
    def forward(self, x): return self.head(self.backbone(x))

def maybe_denorm_xywh(df, img_root, img_size_cache):
    # 如果候选的 w,h 的95分位 <= 2，基本就是归一化到[0,1]——按图像像素还原
    if df.empty: return df
    wh95 = max(df['w'].quantile(0.95), df['h'].quantile(0.95))
    if wh95 > 2.0:
        return df  # 已是像素
    # 逐行按图尺寸放缩
    xs,ys,ws,hs = [],[],[],[]
    for _,r in df.iterrows():
        img = r['image_id']
        if img not in img_size_cache:
            p = os.path.join(img_root, img)
            with Image.open(p) as im:
                img_size_cache[img] = im.size  # (W,H)
        W,H = img_size_cache[img]
        xs.append(float(r['x'])*W); ys.append(float(r['y'])*H)
        ws.append(float(r['w'])*W); hs.append(float(r['h'])*H)
    df = df.copy()
    df['x'],df['y'],df['w'],df['h'] = xs,ys,ws,hs
    return df

def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 1) 读数据 & 统一 image_id
    pred = pd.read_csv(a.pred_csv)
    gt   = pd.read_csv(a.gt_csv).rename(columns={'gx':'x','gy':'y','gw':'w','gh':'h'})

    for df in (pred, gt):
        df['image_id'] = df['image_id'].apply(norm_id)

    # 只评估 pred 涵盖到的图像
    pred_imgs = set(pred['image_id'])
    gt = gt[gt['image_id'].isin(pred_imgs)].copy()

    # 只评估阳性 GT
    gt_pos = gt[gt['icdas']>=1].copy()

    # 2) 加载分级器
    tx = transforms.Compose([
        transforms.Resize((a.img_size,a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    class Model(OrdinalHead): pass
    model = Model().to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()

    # 3) 归一化候选 → 像素（若需要）
    img_size_cache = {}
    pred = pred.copy()
    pred = maybe_denorm_xywh(pred, a.img_root, img_size_cache)

    # 4) 给候选打 p_ge1，并得到融合分
    def crop_tensor(image_id, x,y,w,h, expand=1.15):
        p = os.path.join(a.img_root, image_id)
        with Image.open(p) as im:
            im = im.convert('RGB'); W,H = im.size
            cx,cy = x+w/2, y+h/2; w2,h2 = w*expand, h*expand
            x1=max(0,cx-w2/2); y1=max(0,cy-h2/2); x2=min(W,cx+w2/2); y2=min(H,cy+h2/2)
            return tx(im.crop((x1,y1,x2,y2)))

    ps=[]; batch=[]
    with torch.no_grad():
        for i,r in pred.iterrows():
            batch.append(crop_tensor(r['image_id'], float(r['x']),float(r['y']),float(r['w']),float(r['h']), a.expand))
            if len(batch)==a.bs or i==len(pred)-1:
                x = torch.stack(batch).to(device)
                z = model(x); p = torch.sigmoid(z)[:,0].cpu().numpy()
                ps.extend(p.tolist()); batch=[]
    pred['p_ge1'] = ps
    pred['yolo_score_raw'] = pred['yolo_score']  # 先备份原 YOLO 分

    mode  = os.environ.get('FUSE_MODE', 'sqrt')   # sqrt / alpha / gamma / ponly
    alpha = float(os.environ.get('ALPHA', '0.3')) # alpha 模式下权重
    gamma = float(os.environ.get('GAMMA', '0.8')) # gamma 模式下幂次

    s = pred['yolo_score_raw'].clip(1e-9, 1.0)    # 用“原始分”参与融合
    p = pred['p_ge1'].clip(1e-9, 1.0)

    if mode == 'sqrt':
        pred['score_fused'] = np.sqrt(s * p)
    elif mode == 'alpha':
        pred['score_fused'] = alpha * s + (1 - alpha) * p
    elif mode == 'gamma':
        pred['score_fused'] = s * (p ** gamma)
    elif mode == 'ponly':
        pred['score_fused'] = p
    else:
        pred['score_fused'] = np.sqrt(s * p)

    # 保留基线分，令 NMS 使用融合分
    pred['yolo_score'] = pred['score_fused']

    # 5) 评估（先不做NMS，再做NMS）
    def recall_on(dets):
        hits,total=0,len(gt_pos)
        for img,g in gt_pos.groupby('image_id'):
            cand = dets[dets['image_id']==img]
            for _,r in g.iterrows():
                gbox=[r['x'],r['y'],r['x']+r['w'],r['y']+r['h']]
                ok=False
                for _,c in cand.iterrows():
                    if iou(gbox, to_xyxy_row(c)) >= a.eval_iou:
                        ok=True; break
                hits += int(ok)
        return hits,total,(hits/max(total,1))

    # baseline/fused（无NMS）
    b_hits,b_tot,b_rec = recall_on(pred.rename(columns={'yolo_score':'score'}))
    f_hits,f_tot,f_rec = recall_on(pred.rename(columns={'score_fused':'score'}))

    # baseline/fused（NMS后）
    # baseline/fused（NMS后）——明确指定用哪一列做 NMS 分数
    def do_nms(df, score_col):
        rows=[]
        for img,g in df.groupby('image_id'):
            if len(g)==0: 
                continue
            boxes  = torch.tensor([to_xyxy_row(r) for _,r in g.iterrows()], dtype=torch.float32)
            scores = torch.tensor(g[score_col].values, dtype=torch.float32)
            keep   = nms(boxes, scores, iou_threshold=a.nms_iou).cpu().numpy().tolist()
            rows.append(g.iloc[keep])
        return pd.concat(rows, ignore_index=True) if rows else df.head(0)

    base_nms  = do_nms(pred, 'yolo_score_raw')  # 基线：用原 YOLO 分
    fused_nms = do_nms(pred, 'yolo_score')      # 融合：用覆盖后的融合分

    bN_hits,bN_tot,bN_rec = recall_on(base_nms)
    fN_hits,fN_tot,fN_rec = recall_on(fused_nms)

    # 输出与保存
    fused_nms.to_csv(a.out_csv, index=False)
    print(f"[Sanity] pred images: {len(pred_imgs)} | gt_pos boxes in overlap images: {len(gt_pos)}")
    print(f"Recall@0.5  no-NMS  baseline: {b_hits}/{b_tot}={b_rec:.3f} | fused: {f_hits}/{f_tot}={f_rec:.3f}")
    print(f"Recall@0.5  with-NMS baseline: {bN_hits}/{bN_tot}={bN_rec:.3f} | fused: {fN_hits}/{fN_tot}={fN_rec:.3f}")
    print("Saved fused to:", a.out_csv)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)   # YOLO 低阈值候选 (val)
    ap.add_argument('--img_root', required=True)   # val 图片根目录
    ap.add_argument('--gt_csv', required=True)     # icdas_strong_labels.csv
    ap.add_argument('--ckpt', required=True)       # ordinal_head_resnet18.pt
    ap.add_argument('--out_csv', default='val_predictions_fused.csv')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.15)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--nms_iou', type=float, default=0.6)
    ap.add_argument('--eval_iou', type=float, default=0.5)  # 评估 IoU（默认0.5）
    ap.add_argument('--cand_expand', type=float, default=1.0)  # 仅用于评估时把候选框稍微放大
    a = ap.parse_args(); main(a)
