# train_ordinal_head_min.py
import argparse, os, math, random
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class ROIDataset(Dataset):
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
        x,y,w,h = float(r['x']), float(r['y']), float(r['w']), float(r['h'])
        # 扩边
        cx, cy = x+w/2, y+h/2
        w2, h2 = w*self.expand, h*self.expand
        x1 = max(0, cx - w2/2); y1 = max(0, cy - h2/2)
        x2 = min(W, cx + w2/2); y2 = min(H, cy + h2/2)
        crop = im.crop((x1,y1,x2,y2))
        if self.augment and random.random()<0.5:
            crop = transforms.functional.hflip(crop)
        x = self.tx(crop)
        y_ge = torch.tensor([int(r[f'y_ge{k}']) if not math.isnan(r[f'y_ge{k}']) else 0 for k in range(1,7)], dtype=torch.float32)
        m_ge = torch.tensor([int(r[f'mask_ge{k}']) for k in range(1,7)], dtype=torch.float32)
        return x, y_ge, m_ge

class OrdinalHead(nn.Module):
    def __init__(self, pretrained=True, out_dims=6):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feat = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(feat, out_dims)
    def forward(self, x):
        z = self.head(self.backbone(x))  # (B,6)
        return z

def loss_masked_ordinal(z, y, m, lambda_mono=0.1):
    p = torch.sigmoid(z)
    bce = F.binary_cross_entropy(p, y, reduction='none')
    bce = (bce*m).sum(dim=1) / (m.sum(dim=1)+1e-6)
    # 单调正则：z_{k+1} >= z_k
    mono = F.softplus(z[:,1:] - z[:,:-1]).mean(dim=1)
    return (bce + lambda_mono*mono).mean()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); losses=[]
    for x,y,m in loader:
        x,y,m = x.to(device), y.to(device), m.to(device)
        z = model(x)
        losses.append(loss_masked_ordinal(z,y,m).item())
    return sum(losses)/len(losses)

def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OrdinalHead(pretrained=True).to(device)
    tr = ROIDataset(a.train_csv, a.img_root_train, a.img_size, a.expand, augment=True)
    va = ROIDataset(a.val_csv,   a.img_root_val,   a.img_size, a.expand, augment=False)
    tr_loader = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va, batch_size=a.bs, shuffle=False, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best = 1e9
    for ep in range(a.epochs):
        model.train()
        for x,y,m in tr_loader:
            x,y,m = x.to(device), y.to(device), m.to(device)
            z = model(x)
            loss = loss_masked_ordinal(z,y,m, lambda_mono=a.lambda_mono)
            opt.zero_grad(); loss.backward(); opt.step()
        val_loss = evaluate(model, va_loader, device)
        print(f'Epoch {ep+1}/{a.epochs}  val_loss={val_loss:.4f}')
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), a.out)
    print('best val_loss:', best)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv',   required=True)
    ap.add_argument('--img_root_train', required=True)
    ap.add_argument('--img_root_val',   required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--expand', type=float, default=1.15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lambda_mono', type=float, default=0.1)
    ap.add_argument('--out', default='ordinal_head_resnet18.pt')
    a = ap.parse_args()
    main(a)
