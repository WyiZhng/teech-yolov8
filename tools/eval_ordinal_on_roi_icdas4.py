# tools/eval_ordinal_on_roi_icdas4.py
import os, sys, argparse, numpy as np, pandas as pd, torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ord2seq_head import Ord2SeqOrdinalHead

def norm_id(s): return os.path.basename(str(s)).strip()

class OrdinalHead(torch.nn.Module):
    def __init__(self, out_dims=6):
        super().__init__()
        m = models.resnet18(weights=None)
        feat = m.fc.in_features
        m.fc = torch.nn.Identity()
        self.backbone = m
        self.head = torch.nn.Linear(feat, out_dims)
    def forward(self, x): return self.head(self.backbone(x))


class SoftmaxHead4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=None)
        feat = m.fc.in_features
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.fc = torch.nn.Linear(feat, 4)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(1)
        return self.fc(feat)


class ResNet18Ord2Seq4(torch.nn.Module):
    def __init__(self, d_model=256, num_layers=2, backbone_style='resnet'):
        super().__init__()
        m = models.resnet18(weights=None)
        feat = m.fc.in_features
        self.backbone_style = backbone_style
        if backbone_style == 'sequential':
            self.backbone = nn.Sequential(*list(m.children())[:-1])
        else:
            m.fc = torch.nn.Identity()
            self.backbone = m
        # Keep the same attribute name as training scripts for seamless state_dict loading.
        self.ord_head = Ord2SeqOrdinalHead(
            in_features=feat,
            num_classes=4,
            d_model=d_model,
            nhead=8,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            use_masked_decision=True,
        )

    def forward(self, x):
        feat = self.backbone(x)
        if self.backbone_style == 'sequential':
            feat = feat.flatten(1)
        return self.ord_head(feat)

def map_ic4(icdas):
    if icdas <= 0: return 0
    if icdas <= 2: return 1  # A
    if icdas <= 4: return 2  # B
    return 3                 # C

def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(a.val_csv)
    for c in ['image_id','x','y','w','h','icdas']:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少列: {c}")
    df['image_id'] = df['image_id'].astype(str).apply(norm_id)
    df['ic4'] = df['icdas'].apply(map_ic4)

    # 模型：自动识别 masked-ordinal / softmax4 / ord2seq。
    sd = torch.load(a.ckpt, map_location=device)
    is_ord2seq = any(
        k.startswith('head.step_classifiers')
        or 'head.path_tokens' in k
        or k.startswith('ord_head.step_classifiers')
        or 'ord_head.path_tokens' in k
        for k in sd.keys()
    )
    is_softmax4 = ('fc.weight' in sd and tuple(sd['fc.weight'].shape) == (4, 512))

    ckpt_type = 'ord2seq' if is_ord2seq else ('softmax4' if is_softmax4 else 'masked')

    if ckpt_type == 'ord2seq':
        if 'ord_head.feature_proj.weight' in sd:
            d_model = int(sd['ord_head.feature_proj.weight'].shape[0])
        elif 'head.feature_proj.weight' in sd:
            d_model = int(sd['head.feature_proj.weight'].shape[0])
        else:
            d_model = 256
        # Infer decoder layers from state dict.
        layer_ids = set()
        for k in sd.keys():
            p1 = 'ord_head.decoder.layers.'
            p2 = 'head.decoder.layers.'
            if p1 in k:
                tail = k.split(p1, 1)[1]
            elif p2 in k:
                tail = k.split(p2, 1)[1]
            else:
                tail = None
            if tail is not None:
                idx = tail.split('.', 1)[0]
                if idx.isdigit():
                    layer_ids.add(int(idx))
        num_layers = (max(layer_ids) + 1) if layer_ids else 2
        # Auto-detect backbone state_dict layout:
        # - train_softmax_head_icdas4.py stores sequential keys like backbone.0.weight
        # - train_ordinal_head_min.py (masked branch) style would be backbone.conv1.weight
        if any(k.startswith('backbone.0.') for k in sd.keys()):
            bb_style = 'sequential'
        else:
            bb_style = 'resnet'
        model = ResNet18Ord2Seq4(d_model=d_model, num_layers=num_layers, backbone_style=bb_style).to(device)
    elif ckpt_type == 'softmax4':
        model = SoftmaxHead4().to(device)
    else:
        out_dims = sd['head.weight'].shape[0] if 'head.weight' in sd else 6
        model = OrdinalHead(out_dims=out_dims).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()

    print(f"Detected checkpoint type: {ckpt_type}")
    critical_missing = []
    if ckpt_type == 'ord2seq':
        if any('ord_head.' in k for k in model.state_dict().keys()):
            critical_missing = [k for k in missing if k.startswith('ord_head.')]
        else:
            critical_missing = [k for k in missing if k.startswith('head.')]
    elif ckpt_type == 'softmax4':
        critical_missing = [k for k in missing if k.startswith('fc.')]
    else:
        critical_missing = [k for k in missing if k.startswith('head.')]

    if critical_missing:
        raise SystemExit(
            f"Checkpoint and model head mismatch. Missing critical keys: {critical_missing[:5]}"
        )
    if unexpected:
        print(f"Note: unexpected keys ignored: {len(unexpected)}")
    if missing:
        print(f"Note: missing keys ignored: {len(missing)}")

    tx = transforms.Compose([
        transforms.Resize((a.img_size, a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    def crop_tensor(im_path, x,y,w,h, expand=1.25):
        with Image.open(im_path) as im:
            im = im.convert('RGB')
            W,H = im.size
            cx,cy = x+w/2, y+h/2
            w2,h2 = w*expand, h*expand
            x1 = max(0, cx - w2/2); y1 = max(0, cy - h2/2)
            x2 = min(W, cx + w2/2); y2 = min(H, cy + h2/2)
            return tx(im.crop((x1,y1,x2,y2)))

    # 前向
    ps_ge1, ps_ge3, ps_ge5 = [], [], []
    batch, metas = [], []
    with torch.no_grad():
        for i, r in df.iterrows():
            imgp = os.path.join(a.img_root, r['image_id'])
            batch.append(crop_tensor(imgp, float(r['x']),float(r['y']),float(r['w']),float(r['h']), a.expand))
            metas.append(i)
            if len(batch)==a.bs or i==len(df)-1:
                x = torch.stack(batch).to(device)
                out = model(x)
                if ckpt_type == 'ord2seq':
                    probs = out['prob'].cpu().numpy()  # [B,4]
                    ge1 = 1.0 - probs[:, 0]
                    ge3 = probs[:, 2] + probs[:, 3]
                    ge5 = probs[:, 3]
                elif ckpt_type == 'softmax4':
                    probs = torch.softmax(out, dim=1).cpu().numpy()  # [B,4]
                    ge1 = 1.0 - probs[:, 0]
                    ge3 = probs[:, 2] + probs[:, 3]
                    ge5 = probs[:, 3]
                else:
                    z = out
                    p = torch.sigmoid(z).cpu().numpy()
                    # 兼容 3/6 维输出
                    if p.shape[1] == 3:
                        ge1, ge3, ge5 = p[:,0], p[:,1], p[:,2]
                    else:
                        ge1, ge3, ge5 = p[:,0], p[:,2], p[:,4]
                ps_ge1.extend(ge1.tolist()); ps_ge3.extend(ge3.tolist()); ps_ge5.extend(ge5.tolist())
                batch, metas = [], []

    df['p_ge1'] = ps_ge1
    df['p_ge3'] = ps_ge3
    df['p_ge5'] = ps_ge5

    # --- 阈值层面 AUC ---
    def safe_auc(y, s):
        y = np.asarray(y); s = np.asarray(s)
        if (y==1).sum()>0 and (y==0).sum()>0:
            return roc_auc_score(y, s)
        return float('nan')

    auc_ge1 = safe_auc(df['icdas']>=1, df['p_ge1'])
    auc_ge3 = safe_auc(df['icdas']>=3, df['p_ge3'])
    auc_ge5 = safe_auc(df['icdas']>=5, df['p_ge5'])

    # --- 4 类离散预测（阈值=0.5，可调） ---
    def pred_ic4_row(r, t=0.5):
        if r['p_ge1'] < t: return 0
        if r['p_ge3'] < t: return 1
        if r['p_ge5'] < t: return 2
        return 3
    df['pred_ic4'] = df.apply(pred_ic4_row, axis=1)

    mae = float(np.mean(np.abs(df['pred_ic4'] - df['ic4'])))
    qwk = cohen_kappa_score(df['ic4'], df['pred_ic4'], weights='quadratic')
    cm  = confusion_matrix(df['ic4'], df['pred_ic4'], labels=[0,1,2,3])

    # --- 小预算：按图取 top-K（p_ge1 排序）看是否命中 icdas>=1 的 ROI ---
    def topk_hit(k):
        hits, total = 0, df['image_id'].nunique()
        for img, g in df.groupby('image_id'):
            g = g.sort_values('p_ge1', ascending=False).head(k)
            hits += int((g['icdas']>=1).any())
        return hits, total, hits/max(1,total)

    t1 = topk_hit(1)
    t3 = topk_hit(3)

    # 输出
    print(f"AUC(≥1/≥3/≥5)= {auc_ge1:.3f}/{auc_ge3:.3f}/{auc_ge5:.3f}")
    print(f"4类  MAE={mae:.3f}  QWK={qwk:.3f}")
    print("Confusion Matrix (rows=GT 0/A/B/C, cols=Pred):")
    print(cm)
    print(f"Top-1@p_ge1: {t1[0]}/{t1[1]} = {t1[2]:.3f}")
    print(f"Top-3@p_ge1: {t3[0]}/{t3[1]} = {t3[2]:.3f}")

    if a.out_csv:
        df.to_csv(a.out_csv, index=False)
        print("Saved per-ROI preds to:", a.out_csv)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--img_root', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.25)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--out_csv', default='')
    a = ap.parse_args(); main(a)
