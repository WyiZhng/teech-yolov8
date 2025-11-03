# eval_ordinal_on_roi.py
# 评估：在 ROI 上跑序位模型，计算准确率、MAE、F1 等指标
import os, math, argparse
import pandas as pd
import torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

class OrdinalHead(torch.nn.Module):
    def __init__(self, out_dims=6):
        super().__init__()
        m = models.resnet18(weights=None)  # eval时不需预训练权重
        feat = m.fc.in_features
        m.fc = torch.nn.Identity()
        self.backbone = m
        self.head = torch.nn.Linear(feat, out_dims)
    def forward(self, x): return self.head(self.backbone(x))

def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OrdinalHead().to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()

    tx = transforms.Compose([
        transforms.Resize((a.img_size,a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    df = pd.read_csv(a.val_csv)
    def crop(image_id, x,y,w,h, expand=1.15):
        p = os.path.join(a.img_root, image_id)
        im = Image.open(p).convert('RGB'); W,H = im.size
        cx,cy = x+w/2,y+h/2; w2,h2 = w*expand,h*expand
        x1=max(0,cx-w2/2); y1=max(0,cy-h2/2); x2=min(W,cx+w2/2); y2=min(H,cy+h2/2)
        return tx(im.crop((x1,y1,x2,y2)))

    y_true, y_pred, y_true_bin, y_pred_bin = [], [], [], []
    with torch.no_grad():
        batch, metas = [], []
        for _,r in df.iterrows():
            batch.append(crop(r['image_id'], float(r['x']),float(r['y']),float(r['w']),float(r['h']), a.expand))
            metas.append(int(r['gt_icdas']))
            if len(batch)==a.bs:
                x = torch.stack(batch).to(device)
                z = model(x); p = torch.sigmoid(z).cpu()
                # 序位逆变换：等级 = 满足 p_gek>=0.5 的阈值个数
                lvl = (p>=0.5).sum(dim=1).tolist()
                y_pred += lvl; y_true += metas
                y_pred_bin += (p[:,0]>=0.5).int().tolist()
                y_true_bin += [int(m>=1) for m in metas]
                batch, metas = [], []
        if batch:
            x = torch.stack(batch).to(device)
            z = model(x); p = torch.sigmoid(z).cpu()
            y_pred += (p>=0.5).sum(dim=1).tolist()
            y_true += metas
            y_pred_bin += (p[:,0]>=0.5).int().tolist()
            y_true_bin += [int(m>=1) for m in metas]

    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true==y_pred).mean()
    mae = np.abs(y_true - y_pred).mean()

    tp = ((np.array(y_true_bin)==1)&(np.array(y_pred_bin)==1)).sum()
    fp = ((np.array(y_true_bin)==0)&(np.array(y_pred_bin)==1)).sum()
    fn = ((np.array(y_true_bin)==1)&(np.array(y_pred_bin)==0)).sum()
    prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6); f1 = 2*prec*rec/(prec+rec+1e-6)

    print(f"ROI Val — Acc(exact)={acc:.3f}  MAE={mae:.3f}  F1(≥1)={f1:.3f}  P={prec:.3f}  R={rec:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--img_root', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.15)
    ap.add_argument('--bs', type=int, default=128)
    a = ap.parse_args(); main(a)
