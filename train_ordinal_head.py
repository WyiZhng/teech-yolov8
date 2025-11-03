
# Minimal PyTorch training script for the ordinal head (masked BCE + monotonic regularizer).
# Fill IMG_ROOT_* with your image directories and run.
import argparse, math, os, random
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class ROIDataset(Dataset):
    def __init__(self, csv_path, img_root, img_size=256, augment=True):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.img_size = img_size
        self.augment = augment
        self.id2idx = np.arange(len(self.df))

        self.tx = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img_path = os.path.join(self.img_root, r['image_id'])
        img = Image.open(img_path).convert('RGB')
        x, y, w, h = float(r['x']), float(r['y']), float(r['w']), float(r['h'])
        W, H = img.size
        # clamp box into image
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
        crop = img.crop((x, y, x+w, y+h))

        # (Optional) simple augment
        if self.augment and random.random() < 0.5:
            crop = transforms.functional.hflip(crop)

        crop = self.tx(crop)

        y_ge = []
        mask = []
        for k in range(1,7):
            yv = r.get(f'y_ge{k}', np.nan)
            mv = r.get(f'mask_ge{k}', 0)
            y_ge.append( float(0 if math.isnan(yv) else yv) )
            mask.append( int(mv) )
        y_ge = torch.tensor(y_ge, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return crop, y_ge, mask

class OrdinalHead(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, out_dims=6):
        super().__init__()
        if backbone == 'resnet18':
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise NotImplementedError
        self.head = nn.Linear(feat_dim, out_dims)

    def forward(self, x):
        f = self.backbone(x)
        z = self.head(f)
        return z  # logits (B,6)

def masked_bce_with_monotonicity(logits, y, mask, lambda_mono=0.1):
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy(p, y, reduction='none')
    bce = (bce * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
    mono = F.softplus(logits[:,1:] - logits[:,:-1]).mean(dim=1)
    loss = bce + lambda_mono * mono
    return loss.mean()

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    losses = []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        z = model(x)
        loss = masked_bce_with_monotonicity(z, y, mask, lambda_mono=0.1)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OrdinalHead(backbone='resnet18', pretrained=True, out_dims=6).to(device)
    train_ds = ROIDataset(args.train_csv, args.img_root_train, img_size=args.img_size, augment=True)
    val_ds   = ROIDataset(args.val_csv,   args.img_root_val,   img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            z = model(x)
            loss = masked_bce_with_monotonicity(z, y, mask, lambda_mono=args.lambda_mono)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_loss = eval_epoch(model, val_loader, device)
        print(f'Epoch {ep+1}/{args.epochs} - val_loss={val_loss:.4f}')
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), args.out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, required=True)
    ap.add_argument('--img_root_train', type=str, required=True)  # folder with original images for TRAIN split
    ap.add_argument('--img_root_val', type=str, required=True)    # folder with original images for VAL split
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lambda_mono', type=float, default=0.1)
    ap.add_argument('--out', type=str, default='ordinal_head_resnet18.pt')
    args = ap.parse_args()
    train(args)
