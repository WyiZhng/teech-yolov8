# fuse_infer_and_eval_v2.py
import os
import argparse
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.ops import nms

a = None  # will be set in __main__

def norm_id(s):  # basename 对齐
    return os.path.basename(str(s)).strip()

def to_xyxy_row(r):
    """
    评估阶段的候选框可选扩张：围绕中心按 a.cand_expand 等比放大。
    仅用于 IoU 评估，不影响送入分级器的裁剪区域。
    """
    cx = r['x'] + 0.5 * r['w']
    cy = r['y'] + 0.5 * r['h']
    w  = r['w'] * a.cand_expand
    h  = r['h'] * a.cand_expand
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return [x1, y1, x2, y2]

def iou(a_box, b_box):
    ax1,ay1,ax2,ay2 = a_box
    bx1,by1,bx2,by2 = b_box
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih   = max(0, ix2-ix1), max(0, iy2-iy1)
    inter   = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter / ua

class OrdinalHead(torch.nn.Module):
    def __init__(self, out_dims=6):
        super().__init__()
        m = models.resnet18(weights=None)
        feat = m.fc.in_features
        m.fc = torch.nn.Identity()
        self.backbone = m
        self.head = torch.nn.Linear(feat, out_dims)
    def forward(self, x):
        return self.head(self.backbone(x))

def maybe_denorm_xywh(df, img_root, img_size_cache):
    """
    如果 w/h 的 95 分位 <= 2，认为是归一化坐标，按图像尺寸还原到像素坐标。
    """
    if df.empty:
        return df
    wh95 = max(df['w'].quantile(0.95), df['h'].quantile(0.95))
    if wh95 > 2.0:
        return df  # 已是像素
    xs,ys,ws,hs = [],[],[],[]
    for _,r in df.iterrows():
        img = r['image_id']
        if img not in img_size_cache:
            p = os.path.join(img_root, img)
            with Image.open(p) as im:
                img_size_cache[img] = im.size  # (W,H)
        W,H = img_size_cache[img]
        xs.append(float(r['x']) * W)
        ys.append(float(r['y']) * H)
        ws.append(float(r['w']) * W)
        hs.append(float(r['h']) * H)
    out = df.copy()
    out['x'], out['y'], out['w'], out['h'] = xs, ys, ws, hs
    return out

def soft_nms_gaussian(df, score_col, iou_thr=0.5, sigma=0.5, score_thresh=1e-3):
    """
    简易 Soft-NMS (Gaussian)：返回“打分衰减后仍保留”的行组成的新 DataFrame。
    该实现按图像内逐一处理。
    """
    kept_rows = []
    for img, g in df.groupby('image_id'):
        if len(g) == 0:
            continue
        boxes = np.array([to_xyxy_row(r) for _, r in g.iterrows()], dtype=np.float32)
        scores = g[score_col].to_numpy(dtype=np.float32).copy()
        idxs   = np.arange(len(g))

        selected = []
        while len(idxs) > 0:
            # 选当前最高分
            i = idxs[np.argmax(scores[idxs])]
            selected.append(i)

            # 计算与 i 的 IoU，衰减其余分数
            xi1, yi1, xi2, yi2 = boxes[i]
            x1 = np.maximum(xi1, boxes[idxs,0])
            y1 = np.maximum(yi1, boxes[idxs,1])
            x2 = np.minimum(xi2, boxes[idxs,2])
            y2 = np.minimum(yi2, boxes[idxs,3])
            inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
            area_i = (xi2 - xi1) * (yi2 - yi1)
            area_j = (boxes[idxs,2]-boxes[idxs,0]) * (boxes[idxs,3]-boxes[idxs,1])
            iou_ij = inter / (area_i + area_j - inter + 1e-6)

            decay = np.exp(-(iou_ij * iou_ij) / sigma)
            scores[idxs] = scores[idxs] * np.where(iou_ij > iou_thr, decay, 1.0)

            # 丢弃衰减到阈值以下的
            idxs = idxs[scores[idxs] >= score_thresh]

        kept_rows.append(g.iloc[selected])

    return pd.concat(kept_rows, ignore_index=True) if kept_rows else df.head(0)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) 读数据 & 统一 image_id
    pred = pd.read_csv(args.pred_csv)
    gt   = pd.read_csv(args.gt_csv).rename(columns={'gx':'x','gy':'y','gw':'w','gh':'h'})
    for df in (pred, gt):
        df['image_id'] = df['image_id'].apply(norm_id)

    # 只评估 pred 覆盖到的图像
    pred_imgs = set(pred['image_id'])
    gt = gt[gt['image_id'].isin(pred_imgs)].copy()
    gt_pos = gt[gt['icdas'] >= 1].copy()

    # 2) 加载分级器
    tx = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    model = OrdinalHead().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # 3) 归一化候选 → 像素（若需要）
    img_size_cache = {}
    pred = pred.copy()
    pred = maybe_denorm_xywh(pred, args.img_root, img_size_cache)

    # 4) 给候选打 p_ge1，并得到融合分
    def crop_tensor(image_id, x,y,w,h, expand=1.15):
        pth = os.path.join(args.img_root, image_id)
        with Image.open(pth) as im:
            im = im.convert('RGB'); W,H = im.size
            cx,cy = x + w/2, y + h/2
            w2,h2 = w * expand, h * expand
            x1 = max(0, cx - w2/2); y1 = max(0, cy - h2/2)
            x2 = min(W, cx + w2/2); y2 = min(H, cy + h2/2)
            return tx(im.crop((x1,y1,x2,y2)))

    ps, batch = [], []
    with torch.no_grad():
        for i, r in pred.iterrows():
            batch.append(crop_tensor(r['image_id'], float(r['x']), float(r['y']), float(r['w']), float(r['h']), args.expand))
            if len(batch) == args.bs or i == len(pred) - 1:
                x = torch.stack(batch).to(device)
                z = model(x); p = torch.sigmoid(z)[:,0].cpu().numpy()
                ps.extend(p.tolist()); batch = []
    pred['p_ge1'] = ps

    # 备份原始 YOLO 分
    pred['yolo_score_raw'] = pred['yolo_score'].clip(1e-9, 1.0)

    # 融合模式
    mode  = os.environ.get('FUSE_MODE', 'sqrt')   # sqrt / alpha / gamma / ponly / guard
    alpha = float(os.environ.get('ALPHA', '0.3'))
    gamma = float(os.environ.get('GAMMA', '0.8'))
    lo    = float(os.environ.get('GUARD_LO', '0.05'))
    hi    = float(os.environ.get('GUARD_HI', '0.60'))

    s = pred['yolo_score_raw'].clip(1e-9, 1.0)
    p = pred['p_ge1'].clip(1e-9, 1.0)

    if mode == 'guard':
        score = s.copy()
        mid = (s >= lo) & (s <= hi)
        score[mid] = alpha * s[mid] + (1 - alpha) * p[mid]
        pred['score_fused'] = score
    elif mode == 'sqrt':
        pred['score_fused'] = np.sqrt(s * p)
    elif mode == 'alpha':
        pred['score_fused'] = alpha * s + (1 - alpha) * p
    elif mode == 'gamma':
        pred['score_fused'] = s * (p ** gamma)
    elif mode == 'ponly':
        pred['score_fused'] = p
    else:
        pred['score_fused'] = np.sqrt(s * p)

    # 诊断：Spearman 相关（无需 SciPy）
    if args.print_corr:
        rank_s = pd.Series(s).rank(method='average').to_numpy()
        rank_p = pd.Series(p).rank(method='average').to_numpy()
        rho = np.corrcoef(rank_s, rank_p)[0,1]
        print(f"[Diag] Spearman(s, p) = {rho:.3f}")

    # 覆盖 NMS 使用列为融合分
    pred['yolo_score'] = pred['score_fused']

    # 5) 评估（先不做 NMS，再做 NMS）
    def recall_on(dets):
        # 注意：no-NMS 的 recall 与分数无关（因为不筛选），常与 baseline 相同
        hits, total = 0, len(gt_pos)
        for img, g in gt_pos.groupby('image_id'):
            cand = dets[dets['image_id'] == img]
            for _, r in g.iterrows():
                gbox = [r['x'], r['y'], r['x']+r['w'], r['y']+r['h']]
                ok = False
                for _, c in cand.iterrows():
                    if iou(gbox, to_xyxy_row(c)) >= args.eval_iou:
                        ok = True; break
                hits += int(ok)
        return hits, total, (hits / max(total, 1))

    # baseline / fused（无 NMS）
    b_hits, b_tot, b_rec = recall_on(pred.rename(columns={'yolo_score_raw':'score'}))
    f_hits, f_tot, f_rec = recall_on(pred.rename(columns={'score_fused':'score'}))

    # baseline / fused（NMS 后）
    def do_nms_hard(df, score_col):
        rows = []
        for img, g in df.groupby('image_id'):
            if len(g) == 0:
                continue
            boxes  = torch.tensor([to_xyxy_row(r) for _, r in g.iterrows()], dtype=torch.float32)
            scores = torch.tensor(g[score_col].values, dtype=torch.float32)
            keep   = nms(boxes, scores, iou_threshold=args.nms_iou).cpu().numpy().tolist()
            rows.append(g.iloc[keep])
        return pd.concat(rows, ignore_index=True) if rows else df.head(0)

    def do_nms_soft(df, score_col):
        return soft_nms_gaussian(df, score_col, iou_thr=args.nms_iou, sigma=args.soft_sigma, score_thresh=args.score_thresh)

    base_nms  = do_nms_hard(pred, 'yolo_score_raw') if not args.soft_nms else do_nms_soft(pred.assign(score=pred['yolo_score_raw']), 'score')
    fused_nms = do_nms_hard(pred, 'yolo_score')     if not args.soft_nms else do_nms_soft(pred.assign(score=pred['yolo_score']),     'score')

    bN_hits, bN_tot, bN_rec = recall_on(base_nms)
    fN_hits, fN_tot, fN_rec = recall_on(fused_nms)

    # 输出与保存
    fused_nms.to_csv(args.out_csv, index=False)
    print(f"[Sanity] pred images: {len(pred_imgs)} | gt_pos boxes in overlap images: {len(gt_pos)}")
    print(f"Recall@{args.eval_iou:.2f}  no-NMS  baseline: {b_hits}/{b_tot}={b_rec:.3f} | fused: {f_hits}/{f_tot}={f_rec:.3f}")
    print(f"Recall@{args.eval_iou:.2f}  with-NMS baseline: {bN_hits}/{bN_tot}={bN_rec:.3f} | fused: {fN_hits}/{fN_tot}={fN_rec:.3f}")

    # --- UNION(base ∪ fused)：并集后再做一次轻 NMS ---
    union = pd.concat([
        base_nms.assign(_src='base'),
        fused_nms.assign(_src='fused')
    ], ignore_index=True)

    union['score_union'] = union[['yolo_score_raw', 'yolo_score']].max(axis=1, skipna=True)

    def do_nms_for_union(df):
        rows = []
        for img, g in df.groupby('image_id'):
            if len(g) == 0:
                continue
            boxes  = torch.tensor([to_xyxy_row(r) for _, r in g.iterrows()], dtype=torch.float32)
            scores = torch.tensor(g['score_union'].fillna(0).values, dtype=torch.float32)
            keep   = nms(boxes, scores, iou_threshold=max(0.80, args.nms_iou)).cpu().numpy().tolist()
            rows.append(g.iloc[keep])
        return pd.concat(rows, ignore_index=True) if rows else df.head(0)

    union_nms = do_nms_for_union(union)
    u_hits, u_tot, u_rec = recall_on(union_nms)
    print(f"Recall@{args.eval_iou:.2f}  UNION(base∪fused): {u_hits}/{u_tot}={u_rec:.3f}")

    print("Saved fused to:", args.out_csv)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)          # YOLO 低阈值候选
    ap.add_argument('--img_root', required=True)          # 图像根目录 (split 对应)
    ap.add_argument('--gt_csv', required=True)            # icdas_strong_labels_<split>.csv
    ap.add_argument('--ckpt', required=True)              # ordinal_head_resnet18.pt
    ap.add_argument('--out_csv', default='predictions_fused.csv')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.15)       # ROI 裁剪的上下文（喂分级器）
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--nms_iou', type=float, default=0.6)       # NMS IoU
    ap.add_argument('--eval_iou', type=float, default=0.5)      # 评估 IoU
    ap.add_argument('--cand_expand', type=float, default=1.0)   # 仅评估时候选扩张
    # 诊断
    ap.add_argument('--print_corr', action='store_true', help='打印 Spearman(s, p) 相关性')
    # Soft-NMS 选项
    ap.add_argument('--soft_nms', action='store_true', help='使用 Soft-NMS (Gaussian)')
    ap.add_argument('--soft_sigma', type=float, default=0.5, help='Soft-NMS 高斯 sigma')
    ap.add_argument('--score_thresh', type=float, default=1e-3, help='Soft-NMS 衰减后保留阈值')
    a = ap.parse_args()
    main(a)
