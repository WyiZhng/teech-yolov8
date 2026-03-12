# train_softmax_head_icdas4.py
import argparse, os, math, random
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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
    def __init__(self, pretrained=True):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # 去掉原 fc
        self.fc = nn.Linear(m.fc.in_features, 4)

    def forward(self, x):
        feat = self.backbone(x)   # [B,512,1,1]
        feat = feat.flatten(1)    # [B,512]
        logits = self.fc(feat)    # [B,4]
        return logits

def main(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tr_ds = Icdas4RoiDataset(a.train_csv, a.img_root_train, a.img_size, a.expand, augment=True)
    va_ds = Icdas4RoiDataset(a.val_csv,   a.img_root_val,   a.img_size, a.expand, augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=a.bs, shuffle=True,  num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=a.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet18Softmax4(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(a.epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tr_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)          # [B,4]
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)

        print(f'Epoch {epoch+1}/{a.epochs} train_loss={total_loss/len(tr_ds):.4f}')

        # Val Acc
        model.eval()
        correct, tot = 0, 0
        with torch.no_grad():
            for imgs, labels in va_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                tot += labels.numel()
        print(f'  Val Acc={correct/tot:.3f}')

        torch.save(model.state_dict(), a.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv',   type=str, required=True)
    parser.add_argument('--img_root_train', type=str, required=True)
    parser.add_argument('--img_root_val',   type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--expand',   type=float, default=1.25)
    parser.add_argument('--bs',       type=int, default=64)
    parser.add_argument('--epochs',   type=int, default=60)
    parser.add_argument('--lr',       type=float, default=3e-4)
    parser.add_argument('--out',      type=str, default='softmax_head_icdas4.pt')
    a = parser.parse_args()
    main(a)
