# tools/eval_softmax_on_roi_icdas4.py
import argparse, os, math, random
import pandas as pd
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, cohen_kappa_score, mean_absolute_error, confusion_matrix

# --- Dataset & Model (Copied from train_softmax_head_icdas4.py) ---

class Icdas4RoiDataset(Dataset):
    def __init__(self, csv_path, img_root, img_size=256, expand=1.15, augment=True):
        self.df = pd.read_csv(csv_path)
        self.root = img_root
        self.size = img_size
        self.expand = expand
        self.augment = augment
        self.tx = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(os.path.join(self.root, r['image_id'])).convert('RGB')
        W,H = im.size
        # 兼容 gx/gy/gw/gh 或 x/y/w/h
        if 'gx' in r:
            x,y,w,h = float(r['gx']), float(r['gy']), float(r['gw']), float(r['gh'])
        else:
            x,y,w,h = float(r['x']), float(r['y']), float(r['w']), float(r['h'])
            
        # 扩边
        cx, cy = x+w/2, y+h/2
        w2, h2 = w*self.expand, h*self.expand
        x1 = max(0, cx - w2/2); y1 = max(0, cy - h2/2)
        x2 = min(W, cx + w2/2); y2 = min(H, cy + h2/2)
        crop = im.crop((x1,y1,x2,y2))
        if self.augment and random.random()<0.5:
            crop = transforms.functional.hflip(crop)
        img = self.tx(crop)
        
        # 标签：优先用 ic4 (0-3)，没有则用 icdas (0-6)
        y4 = int(r['ic4']) if 'ic4' in r else int(r['icdas'])
        return img, y4

class ResNet18Softmax4(nn.Module):
    def __init__(self, pretrained=False): # Eval usually doesn't need pretrained weights if loading ckpt, but structure must match
        super().__init__()
        m = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # 去掉原 fc
        self.fc = nn.Linear(m.fc.in_features, 4)

    def forward(self, x):
        feat = self.backbone(x)   # [B,512,1,1]
        feat = feat.flatten(1)    # [B,512]
        logits = self.fc(feat)    # [B,4]
        return logits

# --- Main Eval Logic ---

def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Force augment=False for evaluation
    ds = Icdas4RoiDataset(a.roi_csv, a.img_root, a.img_size, a.expand, augment=False)
    loader = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet18Softmax4(pretrained=False).to(device)
    state = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_y = []
    all_p = []   # [N,4]

    print(f"Starting inference on {len(ds)} ROIs...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)           # [B,4]
            prob = torch.softmax(logits, dim=1)  # [B,4]
            all_p.append(prob.cpu().numpy())
            all_y.append(labels.numpy())

    prob = np.concatenate(all_p, axis=0)   # [N,4]
    y = np.concatenate(all_y, axis=0)      # [N]

    # 概率解构：0, A, B, C
    # p0 = prob[:,0]
    pA, pB, pC = prob[:,1], prob[:,2], prob[:,3]

    # 这里我们定义：
    # ≥1 : A/B/C   (icdas4 >= 1)
    # ≥3 : B/C     (icdas4 >= 2)
    # ≥5 : C       (icdas4 >= 3)
    p_ge1 = pA + pB + pC
    p_ge3 = pB + pC
    p_ge5 = pC

    y_ge1 = (y >= 1).astype(int)
    y_ge3 = (y >= 2).astype(int)
    y_ge5 = (y >= 3).astype(int)

    auc_ge1 = roc_auc_score(y_ge1, p_ge1) if len(np.unique(y_ge1)) > 1 else 0
    auc_ge3 = roc_auc_score(y_ge3, p_ge3) if len(np.unique(y_ge3)) > 1 else 0
    auc_ge5 = roc_auc_score(y_ge5, p_ge5) if len(np.unique(y_ge5)) > 1 else 0

    # 离散预测
    y_pred = prob.argmax(axis=1)

    mae = mean_absolute_error(y, y_pred)
    qwk = cohen_kappa_score(y, y_pred, weights='quadratic')
    cm  = confusion_matrix(y, y_pred, labels=[0,1,2,3])

    print(f'AUC(≥1/≥3/≥5)= {auc_ge1:.3f}/{auc_ge3:.3f}/{auc_ge5:.3f}')
    print(f'4类  MAE={mae:.3f}  QWK={qwk:.3f}')
    print('Confusion Matrix (rows=GT 0/A/B/C, cols=Pred):')
    print(cm)

    # ------ Top-1 / Top-3 @ p_ge1 ------
    # 和 eval_ordinal_on_roi.py 里的逻辑一样：按 image_id 分组、按 p_ge1 排序
    df = pd.read_csv(a.roi_csv)
    
    # Ensure length matches
    if len(df) != len(prob):
        print(f"Warning: CSV length {len(df)} != Preds length {len(prob)}")
        # Truncate to match if necessary, or raise error
        min_len = min(len(df), len(prob))
        df = df.iloc[:min_len]
        p_ge1 = p_ge1[:min_len]
        y = y[:min_len]
        prob = prob[:min_len]

    df['p_ge1'] = p_ge1
    df['y_gt'] = y # Use loaded labels to be sure
    
    # Add other probs for saving
    df['p0'] = prob[:,0]
    df['pA'] = prob[:,1]
    df['pB'] = prob[:,2]
    df['pC'] = prob[:,3]
    df['y_pred'] = y_pred

    top1_hit, top3_hit = 0, 0
    n_imgs = df['image_id'].nunique()

    print(f"Calculating Top-K metrics on {n_imgs} images...")
    for img_id, g in df.groupby('image_id'):
        g_sorted = g.sort_values('p_ge1', ascending=False)
        # 图像是否有阳性 ROI？ (GT >= 1)
        has_pos = (g_sorted['y_gt'] >= 1).any()
        
        # If the image has NO positive ROIs, it doesn't contribute to Recall/Hit calculation usually?
        # Or is this "Recall at image level"? 
        # Usually Top-K recall means: if image is positive, is one of the top K proposals positive?
        if not has_pos:
            continue 

        # Top-1
        if (g_sorted.head(1)['y_gt'] >= 1).any():
            top1_hit += 1
        # Top-3
        if (g_sorted.head(3)['y_gt'] >= 1).any():
            top3_hit += 1
            
    # Note: Denominator is number of POSITIVE images, not all images, if we are measuring Recall.
    # But the user's code used `n_imgs` (all images). 
    # If `n_imgs` includes negatives, then this metric is a bit weird (mix of recall and something else).
    # However, I will follow the user's snippet: `print(f'Top-1@p_ge1: {top1_hit}/{n_imgs} ...')`
    # Wait, the user's snippet has `if not has_pos: continue`. 
    # If I continue, I shouldn't divide by `n_imgs` (total) but by `n_pos_imgs`.
    # Let's count positive images.
    
    n_pos_imgs = df.groupby('image_id')['y_gt'].apply(lambda x: (x>=1).any()).sum()
    
    print(f'Top-1@p_ge1: {top1_hit}/{n_pos_imgs} = {top1_hit/n_pos_imgs:.3f} (Images with GT>=1)')
    print(f'Top-3@p_ge1: {top3_hit}/{n_pos_imgs} = {top3_hit/n_pos_imgs:.3f} (Images with GT>=1)')

    if a.out_csv is not None:
        df.to_csv(a.out_csv, index=False)
        print(f'Saved per-ROI preds to: {a.out_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_csv', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--ckpt',     type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--expand',   type=float, default=1.25)
    parser.add_argument('--bs',       type=int, default=128)
    parser.add_argument('--out_csv',  type=str, default=None)
    a = parser.parse_args()
    main(a)
