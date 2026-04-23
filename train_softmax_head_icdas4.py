# train_softmax_head_icdas4.py
import argparse, os, math, random
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from ord2seq_head import Ord2SeqOrdinalHead


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def map_ic4(icdas):
    if icdas <= 0:
        return 0
    if icdas <= 2:
        return 1
    if icdas <= 4:
        return 2
    return 3

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
        
        # 标签：优先用 ic4 (0-3)；若只有 icdas(0-6) 则映射到 4 类
        if 'ic4' in r:
            y4 = int(r['ic4'])
        else:
            y4 = map_ic4(int(r['icdas']))
        return img, y4


class ResNet18Icdas4(nn.Module):
    def __init__(self, pretrained=True, head_type='softmax', ord2seq_d_model=256, ord2seq_layers=2):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # 去掉原 fc
        self.head_type = head_type
        if self.head_type == 'softmax':
            self.fc = nn.Linear(m.fc.in_features, 4)
        elif self.head_type == 'ord2seq':
            self.ord_head = Ord2SeqOrdinalHead(
                in_features=m.fc.in_features,
                num_classes=4,
                d_model=ord2seq_d_model,
                nhead=8,
                num_decoder_layers=ord2seq_layers,
                dim_feedforward=ord2seq_d_model * 4,
                dropout=0.1,
                use_masked_decision=True,
            )
        else:
            raise ValueError(f'Unknown head_type: {self.head_type}')

    def forward(self, x, labels=None):
        feat = self.backbone(x)   # [B,512,1,1]
        feat = feat.flatten(1)    # [B,512]
        if self.head_type == 'softmax':
            return {'logits': self.fc(feat)}
        return self.ord_head(feat, labels=labels)

def main(a):
    seed_everything(a.seed, deterministic=a.deterministic)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tr_ds = Icdas4RoiDataset(a.train_csv, a.img_root_train, a.img_size, a.expand, augment=True)
    va_ds = Icdas4RoiDataset(a.val_csv,   a.img_root_val,   a.img_size, a.expand, augment=False)

    g = torch.Generator()
    g.manual_seed(a.seed)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=a.bs,
        shuffle=True,
        num_workers=a.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=a.bs,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = ResNet18Icdas4(
        pretrained=True,
        head_type=a.head_type,
        ord2seq_d_model=a.ord2seq_d_model,
        ord2seq_layers=a.ord2seq_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(a.epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tr_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs, labels=labels if a.head_type == 'ord2seq' else None)
            if a.head_type == 'softmax':
                logits = out['logits']
                loss = criterion(logits, labels)
            else:
                loss = out['loss']

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
                out = model(imgs)
                if a.head_type == 'softmax':
                    pred = out['logits'].argmax(dim=1)
                else:
                    pred = out['pred']
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
    parser.add_argument('--head_type', type=str, default='softmax', choices=['softmax', 'ord2seq'])
    parser.add_argument('--ord2seq_d_model', type=int, default=256)
    parser.add_argument('--ord2seq_layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--deterministic', action='store_true')
    a = parser.parse_args()
    main(a)
